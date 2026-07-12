"""Flickr8K caption records and strict CSV parsing.

The original notebook discarded an entire image whenever any reference had
fewer than five words.  That changes the benchmark and biases it toward
verbose captions.  This module preserves every non-empty reference by default;
an explicitly requested minimum length is treated as a validation error rather
than as a silent filtering rule.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    """All human references and the image path for one image."""

    image_id: str
    image_path: Path
    captions: tuple[str, ...]
    normalized_captions: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        image_id = normalize_image_id(self.image_id)
        captions = tuple(_normalize_caption(caption) for caption in self.captions)
        if not captions:
            raise ValueError(f"Image {image_id!r} has no reference captions")
        if len(set(captions)) != len(captions):
            raise ValueError(f"Image {image_id!r} has duplicate reference captions")
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "captions", captions)
        object.__setattr__(
            self,
            "normalized_captions",
            tuple(caption.casefold() for caption in captions),
        )

    @property
    def references(self) -> tuple[str, ...]:
        """Alias that makes the image-level SCST role explicit."""

        return self.captions

    @property
    def normalized_references(self) -> tuple[str, ...]:
        """Case-normalized view used by uncased training rewards."""

        return self.normalized_captions


def normalize_image_id(value: str) -> str:
    """Return a safe basename used consistently by captions and split files."""

    if not isinstance(value, str):
        raise TypeError("image ID must be a string")
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("image ID cannot be empty")
    image_id = PurePosixPath(normalized).name
    if image_id in {"", ".", ".."}:
        raise ValueError(f"Invalid image ID: {value!r}")
    return image_id


def _normalize_caption(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("caption must be a string")
    # Preserve punctuation and case for standards-based caption tokenizers.
    # Only normalize inconsequential whitespace introduced by CSV formatting.
    caption = " ".join(value.split())
    if not caption:
        raise ValueError("reference captions cannot be empty")
    return caption


def validate_records(
    records: Sequence[CaptionRecord] | Iterable[CaptionRecord],
    *,
    expected_references_per_image: int | None = None,
) -> tuple[CaptionRecord, ...]:
    """Validate record identity and reference invariants without reordering."""

    records = tuple(records)
    if not records:
        raise ValueError("caption records cannot be empty")
    if expected_references_per_image is not None and expected_references_per_image < 1:
        raise ValueError("expected_references_per_image must be positive")

    seen: set[str] = set()
    for record in records:
        if not isinstance(record, CaptionRecord):
            raise TypeError("records must contain CaptionRecord instances")
        if record.image_id in seen:
            raise ValueError(f"Duplicate image ID in caption records: {record.image_id!r}")
        seen.add(record.image_id)
        if (
            expected_references_per_image is not None
            and len(record.captions) != expected_references_per_image
        ):
            raise ValueError(
                f"Image {record.image_id!r} has {len(record.captions)} references; "
                f"expected {expected_references_per_image}"
            )
    return records


def _resolve_header(header: list[str], captions_file: Path) -> tuple[int, int]:
    normalized = [item.strip().lstrip("\ufeff").casefold() for item in header]
    image_names = {"image", "image_name", "filename", "file_name"}
    caption_names = {"caption", "description", "text"}
    image_indices = [index for index, name in enumerate(normalized) if name in image_names]
    caption_indices = [index for index, name in enumerate(normalized) if name in caption_names]
    if len(image_indices) != 1 or len(caption_indices) != 1:
        raise ValueError(
            f"{captions_file} must have one image column and one caption column; "
            f"found header {header!r}"
        )
    return image_indices[0], caption_indices[0]


def load_flickr8k_records(
    captions_file: str | Path,
    images_dir: str | Path,
    *,
    min_caption_words: int = 1,
    expected_references_per_image: int | None = 5,
    verify_images: bool = True,
) -> tuple[CaptionRecord, ...]:
    """Parse Flickr8K's ``image,caption`` CSV into ordered image records.

    Properly quoted commas are handled by :mod:`csv`.  Some unofficial mirrors
    contain an unquoted comma inside the final caption field; when the caption
    column is last, the surplus fields are joined back with commas rather than
    silently truncating the reference.

    A caption shorter than ``min_caption_words`` raises a descriptive error.
    No image or caption is silently removed.
    """

    captions_path = Path(captions_file)
    image_root = Path(images_dir)
    if min_caption_words < 1:
        raise ValueError("min_caption_words must be positive")
    if not captions_path.is_file():
        raise FileNotFoundError(f"Caption file does not exist: {captions_path}")
    if verify_images and not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    grouped: dict[str, list[str]] = {}
    with captions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Caption file is empty: {captions_path}") from error
        image_index, caption_index = _resolve_header(header, captions_path)
        required_columns = max(image_index, caption_index) + 1

        for row_number, row in enumerate(reader, start=2):
            if not row or not any(item.strip() for item in row):
                continue
            if len(row) < required_columns:
                raise ValueError(f"Malformed caption row {row_number} in {captions_path}: {row!r}")
            if len(row) > len(header):
                if caption_index != len(header) - 1:
                    raise ValueError(
                        f"Unexpected extra columns on row {row_number} in {captions_path}"
                    )
                row = [*row[:caption_index], ",".join(row[caption_index:])]

            image_id = normalize_image_id(row[image_index])
            try:
                caption = _normalize_caption(row[caption_index])
            except ValueError as error:
                raise ValueError(
                    f"Invalid caption for {image_id!r} on row {row_number}: {error}"
                ) from error
            word_count = len(caption.split())
            if word_count < min_caption_words:
                raise ValueError(
                    f"Caption for {image_id!r} on row {row_number} has {word_count} "
                    f"words; minimum is {min_caption_words}. The record was not deleted."
                )
            grouped.setdefault(image_id, []).append(caption)

    if not grouped:
        raise ValueError(f"Caption file contains no caption rows: {captions_path}")

    records: list[CaptionRecord] = []
    for image_id, captions in grouped.items():
        image_path = image_root / image_id
        if verify_images and not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by captions is missing: {image_path}")
        records.append(CaptionRecord(image_id, image_path, tuple(captions)))

    return validate_records(
        records,
        expected_references_per_image=expected_references_per_image,
    )
