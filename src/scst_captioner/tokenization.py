"""Tokenizer contracts shared by BERT-style and generative tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpecialTokenIds:
    bos: int
    eos: int
    pad: int
    forbidden: tuple[int, ...]


def resolve_special_token_ids(tokenizer: Any) -> SpecialTokenIds:
    """Resolve BOS/EOS, falling back to BERT's CLS/SEP convention."""

    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        bos = getattr(tokenizer, "cls_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "sep_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    if bos is None or eos is None or pad is None:
        raise ValueError("tokenizer must define BOS/CLS, EOS/SEP, and PAD token IDs")
    special_ids = tuple(int(value) for value in getattr(tokenizer, "all_special_ids", ()))
    forbidden = tuple(sorted(set(special_ids) - {int(eos)}))
    return SpecialTokenIds(int(bos), int(eos), int(pad), forbidden)
