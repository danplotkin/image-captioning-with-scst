"""Cross-entropy pretraining with count-weighted metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from scst_captioner.losses import token_cross_entropy_sum


@dataclass(frozen=True)
class XEEpochMetrics:
    loss: float
    accuracy: float
    tokens: int


def _batch_tensor(batch: Mapping[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{name!r}] must be a tensor")
    return value.to(device, non_blocking=True)


class XETrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        pad_token_id: int,
        device: torch.device,
        scheduler: Any | None = None,
        amp: bool = True,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.pad_token_id = pad_token_id
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.global_step = 0

    def train_epoch(self, loader: Iterable[Mapping[str, Any]]) -> XEEpochMetrics:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        for batch in loader:
            pixel_values = _batch_tensor(batch, "pixel_values", self.device)
            input_ids = _batch_tensor(batch, "input_ids", self.device)
            labels = _batch_tensor(batch, "labels", self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                logits = self.model(pixel_values, input_ids)
                loss_sum, correct, count = token_cross_entropy_sum(
                    logits, labels, self.pad_token_id
                )
                if count == 0:
                    raise ValueError("XE batch contains no non-padding target tokens")
                loss = loss_sum / count
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite XE loss: {loss.item()}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            total_loss += float(loss_sum.detach().item())
            total_correct += correct
            total_tokens += count
        if total_tokens == 0:
            raise ValueError("XE training loader produced no target tokens")
        return XEEpochMetrics(
            loss=total_loss / total_tokens,
            accuracy=total_correct / total_tokens,
            tokens=total_tokens,
        )

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Mapping[str, Any]]) -> XEEpochMetrics:
        previous_mode = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        try:
            for batch in loader:
                pixel_values = _batch_tensor(batch, "pixel_values", self.device)
                input_ids = _batch_tensor(batch, "input_ids", self.device)
                labels = _batch_tensor(batch, "labels", self.device)
                logits = self.model(pixel_values, input_ids)
                loss_sum, correct, count = token_cross_entropy_sum(
                    logits, labels, self.pad_token_id
                )
                total_loss += float(loss_sum.item())
                total_correct += correct
                total_tokens += count
        finally:
            self.model.train(previous_mode)
        if total_tokens == 0:
            raise ValueError("XE evaluation loader produced no target tokens")
        return XEEpochMetrics(
            loss=total_loss / total_tokens,
            accuracy=total_correct / total_tokens,
            tokens=total_tokens,
        )
