"""Image-level, multi-reference self-critical sequence training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from scst_captioner.generation import greedy_decode, scst_rollouts
from scst_captioner.losses import reinforce_loss
from scst_captioner.tokenization import resolve_special_token_ids


class Reward(Protocol):
    name: str

    def __call__(
        self, predictions: Sequence[str], references: Sequence[Sequence[str]]
    ) -> Sequence[float] | torch.Tensor: ...


@dataclass(frozen=True)
class SCSTEpochMetrics:
    loss: float
    sampled_reward: float
    baseline_reward: float
    advantage: float
    images: int


def _as_references(value: Any, batch_size: int) -> list[list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("batch['references'] must be a sequence of reference sequences")
    references = [list(item) for item in value]
    if len(references) != batch_size:
        raise ValueError("reference batch size does not match image batch size")
    if any(
        not item or any(not isinstance(reference, str) for reference in item) for item in references
    ):
        raise ValueError("each image must have one or more string references")
    return references


class SCSTTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        tokenizer: Any,
        reward: Reward,
        device: torch.device,
        max_new_tokens: int,
        sample_model_mode: str = "eval",
        include_eos_in_reward: bool = False,
        reward_eos_token: str = "<eos>",
        scheduler: Any | None = None,
        amp: bool = True,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        special_tokens = resolve_special_token_ids(tokenizer)
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.reward = reward
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sample_model_mode = sample_model_mode
        self.include_eos_in_reward = include_eos_in_reward
        self.reward_eos_token = reward_eos_token
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.global_step = 0
        self.special_tokens = special_tokens
        self.forbidden_token_ids = special_tokens.forbidden

    def _decode_for_reward(self, output: Any) -> list[str]:
        decoded = list(
            self.tokenizer.batch_decode(output.token_ids.detach().cpu(), skip_special_tokens=True)
        )
        if not self.include_eos_in_reward:
            return decoded
        actions = output.action_token_ids.detach()
        mask = output.action_mask.detach()
        for index in range(actions.size(0)):
            active_actions = actions[index][mask[index]]
            completed = bool(
                active_actions.numel() and active_actions[-1].item() == self.special_tokens.eos
            )
            if completed:
                decoded[index] = f"{decoded[index]} {self.reward_eos_token}".strip()
        return decoded

    def _references_for_reward(self, references: Sequence[Sequence[str]]) -> list[list[str]]:
        if not self.include_eos_in_reward:
            return [list(values) for values in references]
        return [
            [f"{reference} {self.reward_eos_token}".strip() for reference in values]
            for values in references
        ]

    def _reward_tensor(
        self, predictions: Sequence[str], references: Sequence[Sequence[str]]
    ) -> torch.Tensor:
        values = self.reward(predictions, references)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.shape != (len(predictions),):
            raise ValueError("reward must return exactly one scalar per image")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("reward returned a non-finite value")
        return tensor

    def train_epoch(self, loader: Iterable[Mapping[str, Any]]) -> SCSTEpochMetrics:
        self.model.train()
        totals = {"loss": 0.0, "sample": 0.0, "baseline": 0.0, "advantage": 0.0}
        image_count = 0
        for batch in loader:
            pixel_values = batch.get("pixel_values")
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError("batch['pixel_values'] must be a tensor")
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            batch_size = pixel_values.size(0)
            references = _as_references(batch.get("references"), batch_size)
            image_ids = batch.get("image_ids")
            if image_ids is not None and len(set(image_ids)) != batch_size:
                raise ValueError("SCST batches must contain unique image IDs")

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                rollouts = scst_rollouts(
                    self.model,
                    pixel_values,
                    bos_token_id=self.special_tokens.bos,
                    eos_token_id=self.special_tokens.eos,
                    max_new_tokens=self.max_new_tokens,
                    forbidden_token_ids=self.forbidden_token_ids,
                    sample_model_mode=self.sample_model_mode,
                )
            baseline_text = self._decode_for_reward(rollouts.baseline)
            sampled_text = self._decode_for_reward(rollouts.sampled)
            reward_references = self._references_for_reward(references)
            baseline_reward = self._reward_tensor(baseline_text, reward_references)
            sampled_reward = self._reward_tensor(sampled_text, reward_references)
            advantage = (sampled_reward - baseline_reward).detach()
            if rollouts.sampled.action_log_probs is None:
                raise RuntimeError("sampled rollout did not return action log probabilities")
            loss = reinforce_loss(
                advantage,
                rollouts.sampled.action_log_probs,
                rollouts.sampled.action_mask,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite SCST loss: {loss.item()}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            totals["loss"] += float(loss.detach().item()) * batch_size
            totals["sample"] += float(sampled_reward.sum().item())
            totals["baseline"] += float(baseline_reward.sum().item())
            totals["advantage"] += float(advantage.sum().item())
            image_count += batch_size
        if image_count == 0:
            raise ValueError("SCST training loader produced no images")
        if self.scheduler is not None:
            self.scheduler.step()
        return SCSTEpochMetrics(
            loss=totals["loss"] / image_count,
            sampled_reward=totals["sample"] / image_count,
            baseline_reward=totals["baseline"] / image_count,
            advantage=totals["advantage"] / image_count,
            images=image_count,
        )

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Mapping[str, Any]]) -> float:
        previous_mode = self.model.training
        self.model.eval()
        total_reward = 0.0
        image_count = 0
        try:
            for batch in loader:
                pixel_values = batch.get("pixel_values")
                if not isinstance(pixel_values, torch.Tensor):
                    raise TypeError("batch['pixel_values'] must be a tensor")
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                references = _as_references(batch.get("references"), pixel_values.size(0))
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    output = greedy_decode(
                        self.model,
                        self.model.encode_images(pixel_values),
                        bos_token_id=self.special_tokens.bos,
                        eos_token_id=self.special_tokens.eos,
                        max_new_tokens=self.max_new_tokens,
                        forbidden_token_ids=self.forbidden_token_ids,
                    )
                predictions = self._decode_for_reward(output)
                rewards = self._reward_tensor(predictions, self._references_for_reward(references))
                total_reward += float(rewards.sum().item())
                image_count += pixel_values.size(0)
        finally:
            self.model.train(previous_mode)
        if image_count == 0:
            raise ValueError("SCST evaluation loader produced no images")
        return total_reward / image_count
