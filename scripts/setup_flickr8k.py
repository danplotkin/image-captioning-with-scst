"""Download pinned Flickr8K assets and create the conventional split manifest.

Run from the project root:
    uv run --with kagglehub python scripts/setup_flickr8k.py
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from scst_captioner.config import load_config
from scst_captioner.data import (
    create_official_manifest,
    load_flickr8k_records,
    load_manifest,
    save_manifest,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "adityajn105/flickr8k/versions/1"
SPLIT_ARCHIVE = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
# Digests of the conventional lists and captions validated during setup.
SPLIT_HASHES = {
    "Flickr_8k.trainImages.txt": "e330ddd5156a4c2d2e2b67bc587e70e588f2d132bd8a10dc54dcb71e58115d8e",
    "Flickr_8k.devImages.txt": "18ba829f47f6df1cb830bf7f0506e2ddf333818925299babbd1bbcaf50fdc504",
    "Flickr_8k.testImages.txt": "c0924146c131458ceba45da59670fe9520eadce1e5cafffa16a817672a9b1f17",
}
CAPTIONS_HASH = "e6cf5e8d8025d25b50a4945c38580d96a940b78644f31a90368dfcdca3c7e036"


def ensure_split_lists(directory: Path) -> None:
    """Reuse verified lists or download only missing files, checking before writing."""
    missing = []
    for name, expected in SPLIT_HASHES.items():
        path = directory / name
        if not path.exists():
            missing.append(name)
        elif sha256_file(path) != expected:
            raise ValueError(f"Split list differs from the pinned source: {path}")
    if not missing:
        return
    print("Downloading conventional split lists...", flush=True)
    with (
        urlopen(SPLIT_ARCHIVE, timeout=60) as response,
        ZipFile(BytesIO(response.read())) as archive,
    ):
        payloads = {name: archive.read(name) for name in missing}
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != SPLIT_HASHES[name]:
            raise ValueError(f"Downloaded split list failed SHA256 verification: {name}")
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)


def main() -> None:
    # Config paths and manifest provenance stay consistent regardless of caller cwd.
    os.chdir(PROJECT_ROOT)
    config = load_config("configs/default.toml")
    captions = Path(config.data.captions_file)
    images = Path(config.data.images_dir)
    directory = captions.parent
    if not captions.is_file() or not images.is_dir():
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "Run: uv run --with kagglehub python scripts/setup_flickr8k.py"
            ) from error
        print(f"Downloading {DATASET}...", flush=True)
        kagglehub.dataset_download(DATASET, output_dir=str(directory))
    if sha256_file(captions) != CAPTIONS_HASH:
        raise ValueError(f"Captions differ from the pinned dataset version: {captions}")
    ensure_split_lists(directory)

    print("Validating images, captions, and split assignments...", flush=True)
    records = load_flickr8k_records(
        captions,
        images,
        min_caption_words=config.data.min_caption_words,
        expected_references_per_image=config.data.expected_references_per_image,
        verify_images=True,
    )
    manifest = create_official_manifest(
        records,
        captions_file=captions,
        train_file=directory / "Flickr_8k.trainImages.txt",
        validation_file=directory / "Flickr_8k.devImages.txt",
        test_file=directory / "Flickr_8k.testImages.txt",
        expected_references_per_image=config.data.expected_references_per_image,
        caption_policy=f"minimum-{config.data.min_caption_words}-words-validation-no-filter-v1",
    )
    output = Path(config.data.split_manifest)
    if output.exists():
        if load_manifest(output).to_dict() != manifest.to_dict():
            raise ValueError(
                f"Existing manifest differs from the conventional split or current data: {output}. "
                "Preserve it with any existing experiment artifacts before replacing it."
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
