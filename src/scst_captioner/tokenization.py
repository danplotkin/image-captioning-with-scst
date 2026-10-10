"""Tokenizer contracts shared by BERT-style and generative tokenizers."""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Any

CAPTION_TEXT_DECODING = "explicit_no_hf_cleanup_caption_spacing_v1"
CAPTION_TEXT_NORMALIZATION = "apostrophe_suffixes_interword_hyphen_spacing_v1"
REWARD_TEXT_NORMALIZATION = "wordpiece_punctuation_boundaries_caption_spacing_v1"

_APOSTROPHE_SUFFIX = re.compile(r"(?<=\w)\s*'\s*(s|t|m|d|re|ve|ll)\b(?!\s*')", re.IGNORECASE)
_NEGATION_SUFFIX = re.compile(r"(?<=\w)\s+n'\s*t\b(?!\s*')", re.IGNORECASE)
_INTERWORD_HYPHEN = re.compile(r"(?<=\w)\s*-\s*(?=\w)")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,!?])")


def _join_wordpiece_words(text: str) -> str:
    text = _APOSTROPHE_SUFFIX.sub(r"'\1", text)
    text = _NEGATION_SUFFIX.sub("n't", text)
    return _INTERWORD_HYPHEN.sub("-", text)


def normalize_caption_text(text: str) -> str:
    """Canonicalize spacing lost by WordPiece without joining arbitrary words.

    Standard apostrophe suffixes and single interword hyphens are attached to
    their words. WordPiece cannot distinguish ``tri-colored`` from
    ``tri - colored``; both deliberately use the former spelling. Apply this
    same policy to raw reward references and their document-frequency corpus.

    Arbitrary apostrophes are left alone: joining every `` ' `` would corrupt
    plural possessives such as ``dogs ' tails`` into ``dogs'tails``. Quoted
    single letters and double hyphens are also left alone.
    """

    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", _join_wordpiece_words(text))
    return " ".join(text.split())


def normalize_reward_text(text: str) -> str:
    """Make reward text independent of spacing discarded by WordPiece.

    WordPiece places boundaries around ASCII/Unicode punctuation without
    retaining original whitespace, including internal periods in ``D.C.``
    and ``example.com``. Use these same boundaries for reference text and
    predictions, then rejoin conventional suffixes and single hyphens. This
    is a declared project reward convention, not standard COCO PTB text.
    Apply it to the IDF corpus as well as every scored caption.

    A final completion marker remains an opaque token outside normalization.
    """

    text = text.strip()
    completed = text.casefold().endswith("<eos>")
    if completed:
        text = text[: -len("<eos>")].rstrip()
    text = "".join(
        f" {character} "
        if character in string.punctuation or unicodedata.category(character).startswith("P")
        else character
        for character in text
    )
    normalized = " ".join(_join_wordpiece_words(text).split())
    return f"{normalized} <eos>".strip() if completed else normalized


def decode_caption_batch(tokenizer: Any, token_ids: Any) -> list[str]:
    """Decode generated IDs identically for SCST, evaluation, and inference."""

    # The HF cleanup option has changed defaults between releases and its
    # blanket apostrophe joining corrupts plural possessives. Pin it off and
    # use the declared, narrowly scoped normalization above instead.
    captions = tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [normalize_caption_text(caption) for caption in captions]


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
