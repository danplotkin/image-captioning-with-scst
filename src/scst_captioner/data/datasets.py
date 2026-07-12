"""Separate caption-row XE data from image-level multi-reference SCST data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from scst_captioner.data.records import CaptionRecord, validate_records
from scst_captioner.data.transforms import apply_pil_transform

PILTransform = Callable[[Image.Image], Image.Image]


def _load_rgb_image(record: CaptionRecord, transform: PILTransform | None) -> Image.Image:
    try:
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
    except OSError as error:
        raise OSError(f"Could not load image {record.image_id!r} at {record.image_path}") from error
    return apply_pil_transform(image, transform)


def _process_image(image: Image.Image, image_processor: Any) -> torch.Tensor:
    """Run normalization only after every PIL-space augmentation."""

    processed = image_processor(images=image, return_tensors="pt")
    if not isinstance(processed, Mapping) and not hasattr(processed, "__getitem__"):
        raise TypeError("image processor must return a mapping containing 'pixel_values'")
    try:
        pixel_values = processed["pixel_values"]
    except (KeyError, TypeError) as error:
        raise TypeError("image processor output must contain 'pixel_values'") from error
    if not isinstance(pixel_values, torch.Tensor):
        raise TypeError("image processor 'pixel_values' must be a torch.Tensor")
    if pixel_values.ndim != 4 or pixel_values.size(0) != 1:
        raise ValueError(
            "image processor must return pixel_values shaped [1, channels, height, width]"
        )
    return pixel_values[0]


def _encode_caption(
    tokenizer: Any, caption: str, max_seq_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize to ``max_seq_length + 1`` and then shift for teacher forcing."""

    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least two")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    encoded = tokenizer(
        caption,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length + 1,
        padding=False,
        return_attention_mask=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return a mapping containing 'input_ids'")
    token_ids = encoded["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim == 2 and token_ids.size(0) == 1:
            token_ids = token_ids[0]
        if token_ids.ndim != 1:
            raise ValueError("tokenizer input_ids must represent one sequence")
        ids = token_ids.to(dtype=torch.long)
    else:
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise TypeError("tokenizer input_ids must be a sequence of integers")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids):
            raise TypeError("tokenizer input_ids must contain integers")
        ids = torch.tensor(token_ids, dtype=torch.long)
    if ids.numel() < 2 or ids.numel() > max_seq_length + 1:
        raise ValueError(
            f"tokenizer returned {ids.numel()} IDs; expected between 2 and "
            f"{max_seq_length + 1} after truncation"
        )
    input_ids = ids[:-1]
    labels = ids[1:]
    pad_length = max_seq_length - input_ids.numel()
    input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), value=pad_token_id)
    labels = torch.nn.functional.pad(labels, (0, pad_length), value=pad_token_id)
    return input_ids, labels


class XECaptionDataset(Dataset[dict[str, Any]]):
    """One training example per human caption for teacher-forced XE."""

    def __init__(
        self,
        records: Sequence[CaptionRecord],
        *,
        tokenizer: Any,
        image_processor: Any,
        max_seq_length: int,
        pil_transform: PILTransform | None = None,
    ) -> None:
        self.records = validate_records(records)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_length = max_seq_length
        self.pil_transform = pil_transform
        self._rows = tuple(
            (record_index, caption_index)
            for record_index, record in enumerate(self.records)
            for caption_index in range(len(record.captions))
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, caption_index = self._rows[index]
        record = self.records[record_index]
        caption = record.captions[caption_index]
        image = _load_rgb_image(record, self.pil_transform)
        pixel_values = _process_image(image, self.image_processor)
        input_ids, labels = _encode_caption(
            self.tokenizer,
            caption,
            self.max_seq_length,
        )
        return {
            "caption": caption,
            "caption_index": caption_index,
            "image_id": record.image_id,
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


class SCSTImageDataset(Dataset[dict[str, Any]]):
    """Exactly one example and all references per image for SCST/evaluation."""

    def __init__(
        self,
        records: Sequence[CaptionRecord],
        *,
        image_processor: Any,
        pil_transform: PILTransform | None = None,
        expected_references_per_image: int | None = None,
    ) -> None:
        self.records = validate_records(
            records,
            expected_references_per_image=expected_references_per_image,
        )
        self.image_processor = image_processor
        self.pil_transform = pil_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _load_rgb_image(record, self.pil_transform)
        return {
            "image_id": record.image_id,
            "pixel_values": _process_image(image, self.image_processor),
            "references": record.references,
        }


def collate_image_references(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep references image-major instead of letting default collate transpose them."""

    if not batch:
        raise ValueError("cannot collate an empty SCST batch")
    image_ids: list[str] = []
    references: list[list[str]] = []
    pixels: list[torch.Tensor] = []
    for item in batch:
        image_id = item.get("image_id")
        pixel_values = item.get("pixel_values")
        item_references = item.get("references")
        if not isinstance(image_id, str):
            raise TypeError("SCST item image_id must be a string")
        if image_id in image_ids:
            raise ValueError(f"SCST batch contains duplicate image ID {image_id!r}")
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("SCST item pixel_values must be a tensor")
        if not isinstance(item_references, Sequence) or isinstance(item_references, (str, bytes)):
            raise TypeError("SCST item references must be a sequence of strings")
        if not item_references or any(not isinstance(value, str) for value in item_references):
            raise ValueError("each SCST item must have one or more string references")
        image_ids.append(image_id)
        pixels.append(pixel_values)
        references.append(list(item_references))
    return {
        "image_ids": image_ids,
        "pixel_values": torch.stack(pixels),
        "references": references,
    }


# A descriptive alias for evaluation call sites.
ImageReferenceDataset = SCSTImageDataset
