"""Prepare a small real-image CPU fixture without changing the benchmark split.

Run: uv run python scripts/setup_smoke.py
"""

from __future__ import annotations

import csv
import dataclasses
import os
import shutil
from io import StringIO
from pathlib import Path

from scst_captioner.cli import _load_corpus
from scst_captioner.config import load_config
from scst_captioner.data import (
    FileProvenance,
    create_official_manifest,
    load_flickr8k_records,
    load_manifest,
    records_for_split,
    save_manifest,
    sha256_file,
)

COUNTS = {"train": 16, "validation": 4, "test": 4}


def write_or_verify(path: Path, content: str) -> None:
    payload = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing smoke file differs from the selected subset: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    source = load_config("configs/default.toml")
    config = load_config("configs/smoke.toml")
    records, parent = _load_corpus(source)
    selected = {}
    for split, count in COUNTS.items():
        available = records_for_split(records, parent, split, validate=False)
        if len(available) < count:
            raise ValueError(f"Source {split} split has fewer than {count} images")
        selected[split] = available[:count]

    image_dir = Path(config.data.images_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    captions_path = Path(config.data.captions_file)
    split_paths = {}
    captions = StringIO(newline="")
    writer = csv.writer(captions, lineterminator="\n")
    writer.writerow(["image", "caption"])
    for split, subset in selected.items():
        split_paths[split] = captions_path.parent / f"smoke_{split}Images.txt"
        write_or_verify(split_paths[split], "".join(f"{record.image_id}\n" for record in subset))
        for record in subset:
            destination = image_dir / record.image_id
            if destination.exists():
                if sha256_file(destination) != sha256_file(record.image_path):
                    raise ValueError(f"Existing smoke image differs from source: {destination}")
            else:
                shutil.copyfile(record.image_path, destination)
            writer.writerows((record.image_id, caption) for caption in record.captions)
    write_or_verify(captions_path, captions.getvalue())

    smoke_records = load_flickr8k_records(
        captions_path,
        image_dir,
        min_caption_words=config.data.min_caption_words,
        expected_references_per_image=config.data.expected_references_per_image,
        verify_images=True,
    )
    manifest = create_official_manifest(
        smoke_records,
        captions_file=captions_path,
        train_file=split_paths["train"],
        validation_file=split_paths["validation"],
        test_file=split_paths["test"],
        expected_references_per_image=config.data.expected_references_per_image,
        caption_policy=f"minimum-{config.data.min_caption_words}-words-validation-no-filter-v1",
    )
    manifest = dataclasses.replace(
        manifest,
        algorithm_version="flickr8k-official-prefix-smoke-v1",
        provenance=(
            *manifest.provenance,
            FileProvenance.from_path("parent_split_manifest", source.data.split_manifest),
        ),
    )
    output = Path(config.data.split_manifest)
    if output.exists():
        if load_manifest(output).to_dict() != manifest.to_dict():
            raise ValueError(f"Existing smoke manifest differs from the selected subset: {output}")
        print(f"Verified existing smoke manifest: {output}")
    else:
        save_manifest(manifest, output)
        print(f"Created smoke manifest: {output}")
    print("Images: train=16, validation=4, test=4; XE rows: train=80, validation=20")
    print("Run: uv run scst-captioner train-xe --config configs/smoke.toml")


if __name__ == "__main__":
    main()
