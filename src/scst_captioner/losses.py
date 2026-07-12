"""Supervised and policy-gradient loss helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def reinforce_loss(
    advantage: torch.Tensor,
    action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the SCST sequence loss using an explicit first-EOS mask."""

    if advantage.ndim != 1:
        raise ValueError("advantage must have shape [batch]")
    if action_log_probs.ndim != 2 or action_mask.ndim != 2:
        raise ValueError("action_log_probs and action_mask must have shape [batch, steps]")
    if action_log_probs.shape != action_mask.shape:
        raise ValueError("action_log_probs and action_mask must have identical shapes")
    if action_log_probs.size(0) != advantage.size(0):
        raise ValueError("batch size mismatch between advantage and action log probabilities")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must be boolean")

    sequence_log_prob = action_log_probs.masked_fill(~action_mask, 0.0).sum(dim=1)
    return -(advantage.detach() * sequence_log_prob).mean()


def token_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int
) -> tuple[torch.Tensor, int, int]:
    """Return summed loss, correct-token count, and non-pad-token count."""

    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits leading dimensions must match labels")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=pad_token_id, reduction="sum")
    active = flat_labels.ne(pad_token_id)
    count = int(active.sum().item())
    correct = int((flat_logits.argmax(dim=-1).eq(flat_labels) & active).sum().item())
    return loss, correct, count
