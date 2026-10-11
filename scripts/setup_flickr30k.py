"""Download Flickr30k images and prepare pinned Karpathy captions/splits.

Run: uv run --with kagglehub python scripts/setup_flickr30k.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from scst_captioner.config import load_config
from scst_captioner.data import (
    FileProvenance,
    SplitManifest,
    image_files_sha256,
    load_flickr8k_records,
    load_manifest,
    normalize_image_id,
    records_sha256,
    save_manifest,
    sha256_file,
    validate_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "hsankesara/flickr-image-dataset/versions/1"
ANNOTATION_ARCHIVE = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
ANNOTATION_NAME = "dataset_flickr30k.json"
ANNOTATION_HASH = "db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af"
SPLIT_COUNTS = {"train": 29000, "val": 1014, "test": 1000}


def ensure_annotations(directory: Path) -> Path:
    """Only extract the annotation JSON, verifying its bytes before writing."""
    output = directory / ANNOTATION_NAME
    if output.exists():
        if sha256_file(output) != ANNOTATION_HASH:
            raise ValueError(f"Annotations differ from the pinned source: {output}")
        return output
    print("Downloading Karpathy captions and split assignments...", flush=True)
    with (
        urlopen(ANNOTATION_ARCHIVE, timeout=60) as response,
        ZipFile(BytesIO(response.read())) as archive,
    ):
        payload = archive.read(ANNOTATION_NAME)
    if hashlib.sha256(payload).hexdigest() != ANNOTATION_HASH:
        raise ValueError("Downloaded annotations failed SHA256 verification")
    directory.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return output


def prepare_captions(source: Path, output: Path) -> dict[str, tuple[str, ...]]:
    """Convert raw references to the training CSV without changing split membership."""
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["dataset"] != "flickr30k":
        raise ValueError("Expected Flickr30k annotations")
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_COUNTS}
    seen: set[str] = set()
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["image", "caption"])
    for image in data["images"]:
        image_id = normalize_image_id(image["filename"])
        if image_id != image["filename"] or image_id in seen:
            raise ValueError(f"Unsafe or duplicate annotation image ID: {image['filename']!r}")
        seen.add(image_id)
        if image["split"] not in splits:
            raise ValueError(f"Unknown annotation split: {image['split']!r}")
        splits[image["split"]].append(image_id)
        if len(image["sentences"]) != 5:
            raise ValueError(f"Expected five references for {image_id}")
        for sentence in image["sentences"]:
            caption = sentence["raw"]
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Empty or invalid reference for {image_id}")
            writer.writerow([image_id, caption])
    if {name: len(ids) for name, ids in splits.items()} != SPLIT_COUNTS:
        raise ValueError("Annotation split counts differ from the pinned benchmark")
    payload = stream.getvalue().encode("utf-8")
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError(f"Existing captions differ from the pinned annotations: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    return {name: tuple(ids) for name, ids in splits.items()}


def main() -> None:
    os.chdir(PROJECT_ROOT)
    config = load_config("configs/flickr30k.toml")
    captions = Path(config.data.captions_file)
    images = Path(config.data.images_dir)
    source = ensure_annotations(captions.parent)
    splits = prepare_captions(source, captions)
    if not images.is_dir() or any(
        not (images / image_id).is_file() for ids in splits.values() for image_id in ids
    ):
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "Run: uv run --with kagglehub python scripts/setup_flickr30k.py"
            ) from error
        print(f"Downloading {DATASET}...", flush=True)
        kagglehub.dataset_download(DATASET, output_dir=str(captions.parent))

    print("Validating images, all five references, and split assignments...", flush=True)
    records = load_flickr8k_records(
        captions,
        images,
        min_caption_words=config.data.min_caption_words,
        expected_references_per_image=config.data.expected_references_per_image,
        verify_images=True,
    )
    manifest = SplitManifest(
        dataset="flickr30k",
        strategy="official",
        algorithm_version="flickr30k-karpathy-split-v1",
        caption_policy=f"minimum-{config.data.min_caption_words}-words-validation-no-filter-v1",
        records_sha256=records_sha256(records),
        images_sha256=image_files_sha256(records),
        expected_references_per_image=config.data.expected_references_per_image,
        train=splits["train"],
        validation=splits["val"],
        test=splits["test"],
        provenance=(
            FileProvenance.from_path("captions", captions),
            FileProvenance.from_path("karpathy_annotations", source),
        ),
    )
    validate_manifest(manifest, records)
    output = Path(config.data.split_manifest)
    if output.exists():
        if load_manifest(output).to_dict() != manifest.to_dict():
            raise ValueError(
                f"Existing manifest differs from the Karpathy split or current data: {output}. "
                "Preserve it with existing experiment artifacts before replacing it."
            )
        print(f"Existing manifest verified: {output}")
    else:
        save_manifest(manifest, output)
        print(f"Created manifest: {output}")
    print(
        f"train={len(manifest.train)}, validation={len(manifest.validation)}, "
        f"test={len(manifest.test)}, excluded={len(manifest.excluded)}"
    )


if __name__ == "__main__":
    main()
