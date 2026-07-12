"""Flickr8K records, deterministic splits, and task-specific datasets."""

from scst_captioner.data.datasets import (
    ImageReferenceDataset,
    SCSTImageDataset,
    XECaptionDataset,
    collate_image_references,
)
from scst_captioner.data.records import (
    CaptionRecord,
    load_flickr8k_records,
    normalize_image_id,
    validate_records,
)
from scst_captioner.data.splits import (
    FileProvenance,
    SplitManifest,
    create_official_manifest,
    create_seeded_manifest,
    image_files_sha256,
    load_manifest,
    read_official_split,
    records_for_split,
    records_sha256,
    save_manifest,
    sha256_file,
    validate_manifest,
)
from scst_captioner.data.transforms import apply_pil_transform, build_train_augmentation

__all__ = [
    "CaptionRecord",
    "FileProvenance",
    "ImageReferenceDataset",
    "SCSTImageDataset",
    "SplitManifest",
    "XECaptionDataset",
    "apply_pil_transform",
    "build_train_augmentation",
    "collate_image_references",
    "create_official_manifest",
    "create_seeded_manifest",
    "image_files_sha256",
    "load_flickr8k_records",
    "load_manifest",
    "normalize_image_id",
    "read_official_split",
    "records_for_split",
    "records_sha256",
    "save_manifest",
    "sha256_file",
    "validate_manifest",
    "validate_records",
]
