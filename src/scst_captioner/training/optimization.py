"""Optimizer updates shared by supervised and policy-gradient training."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from scst_captioner.config import XEConfig


def step_optimizer(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: Iterable[nn.Parameter],
    *,
    gradient_clip_norm: float,
) -> bool:
    """Clip unscaled gradients and report whether an optimizer update occurred.

    AMP overflow is handled by GradScaler: it skips the optimizer update and
    lowers its scale. Full-precision nonfinite gradients remain an error.
    """

    parameters = list(parameters)
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        gradient_clip_norm,
        error_if_nonfinite=not scaler.is_enabled(),
    )
    # The norm reduction can overflow even when every gradient is finite.
    # GradScaler only detects nonfinite gradient elements, so it cannot
    # skip this case; fail before an update with gradients zeroed by clipping.
    if (
        scaler.is_enabled()
        and not torch.isfinite(total_norm)
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
    ):
        raise RuntimeError("The total gradient norm is non-finite despite finite AMP gradients")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # A successful step keeps or grows the scale; overflow reduces it.
    return scaler.get_scale() >= previous_scale


def build_xe_optimizer(model: nn.Module, config: XEConfig) -> torch.optim.AdamW:
    """Preserve the legacy single group unless an encoder LR is specified."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config.encoder_learning_rate is None:
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    encoder_ids = {id(p) for p in model.backbone.parameters()}
    groups = []
    for is_encoder, rate in ((False, config.learning_rate), (True, config.encoder_learning_rate)):
        selected = [p for p in parameters if (id(p) in encoder_ids) == is_encoder]
        if selected:
            groups.append({"params": selected, "lr": rate})
    return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
