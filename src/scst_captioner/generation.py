"""Shape-stable greedy, stochastic, and beam decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.distributions import Categorical


class AutoregressiveCaptioner(Protocol):
    training: bool
    encoder_is_frozen: bool

    def train(self, mode: bool = True) -> AutoregressiveCaptioner: ...

    def eval(self) -> AutoregressiveCaptioner: ...

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

    def predict_next_token(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class GenerationOutput:
    """Generated tokens and policy actions.

    ``token_ids`` includes BOS and has shape ``[B, 1 + S]``. Log
    probabilities and ``action_mask`` describe the ``S`` generated actions.
    The mask includes the first EOS action and excludes every later action.
    """

    token_ids: torch.Tensor
    action_log_probs: torch.Tensor | None
    action_mask: torch.Tensor
    sequence_log_probs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.action_mask.ndim != 2:
            raise ValueError("generation tensors must retain [batch, time] dimensions")
        expected = (self.token_ids.size(0), self.token_ids.size(1) - 1)
        if tuple(self.action_mask.shape) != expected:
            raise ValueError("action_mask must correspond to tokens after BOS")
        if self.action_mask.dtype != torch.bool:
            raise TypeError("action_mask must be boolean")
        if self.action_log_probs is not None and self.action_log_probs.shape != expected:
            raise ValueError("action_log_probs must correspond to tokens after BOS")

    @property
    def action_token_ids(self) -> torch.Tensor:
        return self.token_ids[:, 1:]


@dataclass(frozen=True)
class SCSTRollouts:
    baseline: GenerationOutput
    sampled: GenerationOutput


def forbidden_special_ids(
    all_special_ids: Sequence[int], *, bos_token_id: int, eos_token_id: int
) -> tuple[int, ...]:
    """Forbid non-semantic special actions while retaining EOS."""

    del bos_token_id  # BOS is already present in all_special_ids for normal tokenizers.
    return tuple(sorted({int(token_id) for token_id in all_special_ids} - {eos_token_id}))


def _mask_logits(logits: torch.Tensor, forbidden_token_ids: Sequence[int]) -> torch.Tensor:
    if not forbidden_token_ids:
        return logits
    valid_ids = [index for index in forbidden_token_ids if 0 <= index < logits.size(-1)]
    if not valid_ids:
        return logits
    masked = logits.clone()
    masked[:, valid_ids] = -torch.inf
    if torch.isneginf(masked).all(dim=-1).any():
        raise ValueError("The forbidden-token policy masked the complete vocabulary")
    return masked


def _initial_state(memory: torch.Tensor, bos_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = memory.size(0)
    device = memory.device
    tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return tokens, finished


def greedy_decode(
    model: AutoregressiveCaptioner,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    forbidden_token_ids: Sequence[int] = (),
) -> GenerationOutput:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    tokens, finished = _initial_state(memory, bos_token_id)
    masks: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        active = ~finished
        logits = _mask_logits(model.predict_next_token(tokens, memory), forbidden_token_ids)
        next_tokens = logits.argmax(dim=-1)
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, eos_token_id))
        masks.append(active)
        tokens = torch.cat((tokens, next_tokens[:, None]), dim=1)
        finished = finished | next_tokens.eq(eos_token_id)
        if finished.all():
            break

    action_mask = torch.stack(masks, dim=1)
    return GenerationOutput(tokens, None, action_mask)


def sample_decode(
    model: AutoregressiveCaptioner,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    forbidden_token_ids: Sequence[int] = (),
    temperature: float = 1.0,
) -> GenerationOutput:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    tokens, finished = _initial_state(memory, bos_token_id)
    masks: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        active = ~finished
        logits = model.predict_next_token(tokens, memory) / temperature
        distribution = Categorical(logits=_mask_logits(logits, forbidden_token_ids))
        next_tokens = distribution.sample()
        action_log_prob = distribution.log_prob(next_tokens).masked_fill(~active, 0.0)
        next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, eos_token_id))
        masks.append(active)
        log_probs.append(action_log_prob)
        tokens = torch.cat((tokens, next_tokens[:, None]), dim=1)
        finished = finished | next_tokens.eq(eos_token_id)
        if finished.all():
            break

    action_mask = torch.stack(masks, dim=1)
    action_log_probs = torch.stack(log_probs, dim=1)
    return GenerationOutput(tokens, action_log_probs, action_mask)


def length_penalty(action_length: int, alpha: float) -> float:
    """Google-NMT-style length penalty; raw sequence log-probability is unchanged."""

    if action_length < 1:
        raise ValueError("action_length must be positive")
    if alpha < 0:
        raise ValueError("alpha cannot be negative")
    return ((5.0 + action_length) / 6.0) ** alpha


@dataclass(frozen=True)
class _Beam:
    tokens: torch.Tensor
    raw_log_prob: float
    complete: bool

    def normalized_score(self, alpha: float) -> float:
        return self.raw_log_prob / length_penalty(self.tokens.numel() - 1, alpha)


def _beam_decode_one(
    model: AutoregressiveCaptioner,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    beam_size: int,
    length_penalty_alpha: float,
    forbidden_token_ids: Sequence[int],
) -> _Beam:
    device = memory.device
    initial = torch.tensor([bos_token_id], dtype=torch.long, device=device)
    beams = [_Beam(initial, 0.0, False)]

    for _ in range(max_new_tokens):
        candidates: list[_Beam] = []
        for beam in beams:
            if beam.complete:
                candidates.append(beam)
                continue
            logits = model.predict_next_token(beam.tokens[None, :], memory)
            log_probs = _mask_logits(logits, forbidden_token_ids).log_softmax(dim=-1)
            values, indices = log_probs.topk(min(beam_size, log_probs.size(-1)), dim=-1)
            for value, index in zip(values[0], indices[0], strict=True):
                token_id = int(index.item())
                candidates.append(
                    _Beam(
                        torch.cat((beam.tokens, index.reshape(1))),
                        beam.raw_log_prob + float(value.item()),
                        token_id == eos_token_id,
                    )
                )
        beams = sorted(
            candidates,
            key=lambda item: item.normalized_score(length_penalty_alpha),
            reverse=True,
        )[:beam_size]
        if all(beam.complete for beam in beams):
            break
    return max(beams, key=lambda item: item.normalized_score(length_penalty_alpha))


def beam_decode(
    model: AutoregressiveCaptioner,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    beam_size: int,
    length_penalty_alpha: float = 0.0,
    forbidden_token_ids: Sequence[int] = (),
) -> GenerationOutput:
    if max_new_tokens < 1 or beam_size < 1:
        raise ValueError("max_new_tokens and beam_size must be positive")
    beams = [
        _beam_decode_one(
            model,
            memory[index : index + 1],
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            beam_size=beam_size,
            length_penalty_alpha=length_penalty_alpha,
            forbidden_token_ids=forbidden_token_ids,
        )
        for index in range(memory.size(0))
    ]
    max_length = max(beam.tokens.numel() for beam in beams)
    token_ids = torch.full(
        (len(beams), max_length), eos_token_id, dtype=torch.long, device=memory.device
    )
    action_mask = torch.zeros((len(beams), max_length - 1), dtype=torch.bool, device=memory.device)
    raw_scores = torch.empty(len(beams), dtype=torch.float32, device=memory.device)
    for index, beam in enumerate(beams):
        token_ids[index, : beam.tokens.numel()] = beam.tokens
        action_mask[index, : beam.tokens.numel() - 1] = True
        raw_scores[index] = beam.raw_log_prob
    return GenerationOutput(token_ids, None, action_mask, raw_scores)


def scst_rollouts(
    model: AutoregressiveCaptioner,
    pixel_values: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    forbidden_token_ids: Sequence[int] = (),
    sample_model_mode: str = "eval",
) -> SCSTRollouts:
    """Generate an inference-mode baseline and a differentiable sample.

    The caller's original module mode is restored. Frozen encoders are encoded
    once. A trainable encoder is encoded once for eval-mode sampling and twice
    when train-mode sampling is requested, because the two passes deliberately
    use different dropout modes.
    """

    if sample_model_mode not in {"train", "eval"}:
        raise ValueError("sample_model_mode must be 'train' or 'eval'")
    previous_mode = model.training
    try:
        model.eval()
        encoder_frozen = bool(getattr(model, "encoder_is_frozen", False))
        if encoder_frozen:
            with torch.no_grad():
                memory = model.encode_images(pixel_values)
                baseline = greedy_decode(
                    model,
                    memory,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                    forbidden_token_ids=forbidden_token_ids,
                )
            if sample_model_mode == "train":
                model.train()
            sampled = sample_decode(
                model,
                memory,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                forbidden_token_ids=forbidden_token_ids,
            )
        elif sample_model_mode == "eval":
            memory = model.encode_images(pixel_values)
            with torch.no_grad():
                baseline = greedy_decode(
                    model,
                    memory.detach(),
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                    forbidden_token_ids=forbidden_token_ids,
                )
            sampled = sample_decode(
                model,
                memory,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                forbidden_token_ids=forbidden_token_ids,
            )
        else:
            with torch.no_grad():
                baseline_memory = model.encode_images(pixel_values)
                baseline = greedy_decode(
                    model,
                    baseline_memory,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                    forbidden_token_ids=forbidden_token_ids,
                )
            model.train()
            sample_memory = model.encode_images(pixel_values)
            sampled = sample_decode(
                model,
                sample_memory,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                forbidden_token_ids=forbidden_token_ids,
            )
        return SCSTRollouts(baseline=baseline, sampled=sampled)
    finally:
        model.train(previous_mode)
