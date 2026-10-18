"""Shape-stable greedy, stochastic, and beam decoding."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch.distributions import Categorical


class AutoregressiveCaptioner(Protocol):
    training: bool
    encoder_is_frozen: bool

    def train(self, mode: bool = True) -> AutoregressiveCaptioner: ...

    def eval(self) -> AutoregressiveCaptioner: ...

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

    def predict_next_token(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor: ...


class CachedAutoregressiveCaptioner(AutoregressiveCaptioner, Protocol):
    """Optional incremental interface; older captioners use full prefixes."""

    def init_decode_cache(self, memory: torch.Tensor) -> Any: ...

    def decode_step(self, token_ids: torch.Tensor, cache: Any) -> tuple[torch.Tensor, Any]: ...


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
    baseline: GenerationOutput | None
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


def _decode_cache(model: AutoregressiveCaptioner, memory: torch.Tensor, enabled: bool) -> Any:
    # Full-prefix train-mode dropout redraws old states each step. Preserve that
    # policy by using the original path whenever the model is in train mode.
    if enabled and not model.training and hasattr(model, "init_decode_cache"):
        return cast(CachedAutoregressiveCaptioner, model).init_decode_cache(memory)
    return None


def _next_logits(
    model: AutoregressiveCaptioner, tokens: torch.Tensor, memory: torch.Tensor, cache: Any
) -> tuple[torch.Tensor, Any]:
    if cache is None:
        return model.predict_next_token(tokens, memory), None
    return cast(CachedAutoregressiveCaptioner, model).decode_step(tokens[:, -1:], cache)


def greedy_decode(
    model: AutoregressiveCaptioner,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    forbidden_token_ids: Sequence[int] = (),
    use_kv_cache: bool = True,
) -> GenerationOutput:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    tokens, finished = _initial_state(memory, bos_token_id)
    cache = _decode_cache(model, memory, use_kv_cache)
    masks: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        active = ~finished
        logits, cache = _next_logits(model, tokens, memory, cache)
        logits = _mask_logits(logits, forbidden_token_ids)
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
    use_kv_cache: bool = True,
) -> GenerationOutput:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    tokens, finished = _initial_state(memory, bos_token_id)
    cache = _decode_cache(model, memory, use_kv_cache)
    masks: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        active = ~finished
        logits, cache = _next_logits(model, tokens, memory, cache)
        logits = logits / temperature
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
    cache: Any = None

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
    use_kv_cache: bool,
) -> _Beam:
    device = memory.device
    initial = torch.tensor([bos_token_id], dtype=torch.long, device=device)
    beams = [_Beam(initial, 0.0, False, _decode_cache(model, memory, use_kv_cache))]

    for _ in range(max_new_tokens):
        candidates: list[_Beam] = []
        for beam in beams:
            if beam.complete:
                candidates.append(beam)
                continue
            logits, cache = _next_logits(model, beam.tokens[None, :], memory, beam.cache)
            log_probs = _mask_logits(logits, forbidden_token_ids).log_softmax(dim=-1)
            values, indices = log_probs.topk(min(beam_size, log_probs.size(-1)), dim=-1)
            for value, index in zip(values[0], indices[0], strict=True):
                token_id = int(index.item())
                candidates.append(
                    _Beam(
                        torch.cat((beam.tokens, index.reshape(1))),
                        beam.raw_log_prob + float(value.item()),
                        token_id == eos_token_id,
                        # Siblings share an immutable prefix; decode_step returns new state.
                        cache,
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
    use_kv_cache: bool = True,
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
            use_kv_cache=use_kv_cache,
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
    use_kv_cache: bool = True,
    baseline: str = "greedy",
    amp: bool = False,
) -> SCSTRollouts:
    """Generate a differentiable sample and an optional no-grad greedy baseline.

    Owns separate, non-overlapping autocast scopes with weight caching enabled.
    Do not wrap this function in autocast: exiting an inner scope would not clear
    the outer scope's detached baseline weight casts. Pass amp=True instead.
    Frozen encoders and eval-mode trainable encoders are encoded once; train-mode
    trainable encoders are encoded separately for the eval-mode baseline.
    The caller's module mode is restored, including when generation fails.
    """
    if sample_model_mode not in {"train", "eval"}:
        raise ValueError("sample_model_mode must be 'train' or 'eval'")
    if baseline not in {"greedy", "none"}:
        raise ValueError("baseline must be 'greedy' or 'none'")
    device_type = pixel_values.device.type
    if torch.is_autocast_enabled(device_type):
        raise ValueError("scst_rollouts owns autocast scopes; pass amp=True without outer autocast")

    def autocast():
        return torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=amp, cache_enabled=True
        )

    options = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        forbidden_token_ids=forbidden_token_ids,
        use_kv_cache=use_kv_cache,
    )
    previous_mode = model.training
    try:
        model.eval()
        encoder_frozen = bool(getattr(model, "encoder_is_frozen", False))
        shared_memory = None
        if encoder_frozen or sample_model_mode == "eval":
            # Preserve encoder sharing and its gradient graph. This scope exits
            # before either decoder rollout; only the baseline gets a detach.
            with autocast(), torch.no_grad() if encoder_frozen else nullcontext():
                shared_memory = model.encode_images(pixel_values)

        baseline_output = None
        if baseline == "greedy":
            with autocast(), torch.no_grad():
                baseline_memory = (
                    shared_memory.detach()
                    if shared_memory is not None
                    else model.encode_images(pixel_values)
                )
                baseline_output = greedy_decode(model, baseline_memory, **options)
        # Exiting the baseline's outermost scope drops its detached FP16 casts.
        model.train(sample_model_mode == "train")
        with autocast():
            sample_memory = (
                shared_memory if shared_memory is not None else model.encode_images(pixel_values)
            )
            sampled = sample_decode(model, sample_memory, **options)
        return SCSTRollouts(baseline=baseline_output, sampled=sampled)
    finally:
        model.train(previous_mode)
