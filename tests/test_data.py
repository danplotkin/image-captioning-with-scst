from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image

from scst_captioner.data import (
    CaptionRecord,
    SCSTImageDataset,
    SplitManifest,
    XECaptionDataset,
    collate_image_references,
    create_official_manifest,
    create_seeded_manifest,
    load_flickr8k_records,
    load_manifest,
    read_official_split,
    records_for_split,
    save_manifest,
    sha256_file,
    validate_manifest,
    validate_records,
)
from scst_captioner.data.transforms import apply_pil_transform


class FakeImageProcessor:
    def __init__(self) -> None:
        self.seen_pixels: list[tuple[int, int, int]] = []

    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
        assert isinstance(images, Image.Image)
        assert return_tensors == "pt"
        pixel = images.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        self.seen_pixels.append(pixel)
        value = pixel[0] / 255.0
        return {"pixel_values": torch.full((1, 3, 2, 2), value)}


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 101
    eos_token_id = 102

    def __call__(
        self,
        caption: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        padding: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens
        assert truncation
        assert padding is False
        assert not return_attention_mask
        word_ids = [200 + index for index, _ in enumerate(caption.split())]
        word_ids = word_ids[: max_length - 2]
        ids = [self.bos_token_id, *word_ids, self.eos_token_id]
        return {"input_ids": ids}


def _write_image(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    Image.new("RGB", (3, 3), color).save(path)


def _records(tmp_path: Path, count: int, references: int = 2) -> tuple[CaptionRecord, ...]:
    result = []
    for index in range(count):
        image_id = f"image-{index}.jpg"
        _write_image(tmp_path / image_id, (index, index, index))
        result.append(
            CaptionRecord(
                image_id=image_id,
                image_path=tmp_path / image_id,
                captions=tuple(
                    f"reference {reference_index} for image {index}"
                    for reference_index in range(references)
                ),
            )
        )
    return tuple(result)


def test_csv_parser_preserves_short_and_comma_captions(tmp_path: Path) -> None:
    images_dir = tmp_path / "Images"
    images_dir.mkdir()
    _write_image(images_dir / "one.jpg")
    captions_file = tmp_path / "captions.txt"
    captions_file.write_text(
        "image,caption\none.jpg,Dog.\none.jpg,A dog, running beside a person.\n",
        encoding="utf-8",
    )

    records = load_flickr8k_records(
        captions_file,
        images_dir,
        expected_references_per_image=2,
    )

    assert [record.image_id for record in records] == ["one.jpg"]
    assert records[0].captions == ("Dog.", "A dog, running beside a person.")

    with pytest.raises(ValueError, match="record was not deleted"):
        load_flickr8k_records(
            captions_file,
            images_dir,
            min_caption_words=2,
            expected_references_per_image=2,
        )


def test_csv_parser_validates_duplicate_and_reference_counts(tmp_path: Path) -> None:
    images_dir = tmp_path / "Images"
    images_dir.mkdir()
    _write_image(images_dir / "one.jpg")
    captions_file = tmp_path / "captions.txt"
    captions_file.write_text(
        "image,caption\none.jpg,The same caption.\none.jpg,The same caption.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate reference"):
        load_flickr8k_records(
            captions_file,
            images_dir,
            expected_references_per_image=2,
        )

    captions_file.write_text("image,caption\none.jpg,Only one caption.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2"):
        load_flickr8k_records(
            captions_file,
            images_dir,
            expected_references_per_image=2,
        )


def test_xe_and_scst_datasets_have_distinct_granularity_and_transform_order(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path / "first.jpg")
    _write_image(tmp_path / "second.jpg", (20, 20, 20))
    records = (
        CaptionRecord(
            "first.jpg",
            tmp_path / "first.jpg",
            ("first caption", "second human caption"),
        ),
        CaptionRecord(
            "second.jpg",
            tmp_path / "second.jpg",
            ("another caption", "another human reference"),
        ),
    )
    processor = FakeImageProcessor()

    def whiten_before_normalization(image: Image.Image) -> Image.Image:
        assert isinstance(image, Image.Image)
        return Image.new("RGB", image.size, (255, 255, 255))

    xe_dataset = XECaptionDataset(
        records,
        tokenizer=FakeTokenizer(),
        image_processor=processor,
        max_seq_length=4,
        pil_transform=whiten_before_normalization,
    )
    assert len(xe_dataset) == 4
    xe_item = xe_dataset[0]
    assert xe_item["image_id"] == "first.jpg"
    assert xe_item["input_ids"].tolist() == [101, 200, 201, 0]
    assert xe_item["labels"].tolist() == [200, 201, 102, 0]
    assert xe_item["pixel_values"].shape == (3, 2, 2)
    assert processor.seen_pixels[-1] == (255, 255, 255)

    scst_dataset = SCSTImageDataset(
        records,
        image_processor=processor,
        pil_transform=whiten_before_normalization,
        expected_references_per_image=2,
    )
    assert len(scst_dataset) == 2
    first = scst_dataset[0]
    second = scst_dataset[1]
    assert first["references"] == records[0].captions
    batch = collate_image_references([first, second])
    assert batch["image_ids"] == ["first.jpg", "second.jpg"]
    assert batch["references"] == [list(records[0].captions), list(records[1].captions)]
    assert batch["pixel_values"].shape == (2, 3, 2, 2)


def test_pil_transform_rejects_premature_tensor_conversion() -> None:
    image = Image.new("RGB", (2, 2))
    with pytest.raises(TypeError, match=r"must return a PIL\.Image"):
        apply_pil_transform(image, lambda _: torch.zeros(3, 2, 2))  # type: ignore[arg-type,return-value]


def test_default_train_augmentation_is_spawn_worker_picklable() -> None:
    from scst_captioner.data import build_train_augmentation

    pickle.loads(pickle.dumps(build_train_augmentation()))


def test_seeded_manifest_is_deterministic_ordered_and_provenance_checked(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path, 10)
    captions_file = tmp_path / "captions.txt"
    captions_file.write_text("stable caption source\n", encoding="utf-8")

    first = create_seeded_manifest(
        records,
        captions_file=captions_file,
        seed=17,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        expected_references_per_image=2,
    )
    second = create_seeded_manifest(
        tuple(reversed(records)),
        captions_file=captions_file,
        seed=17,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        expected_references_per_image=2,
    )

    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (6, 2, 2)
    assert (first.train_fraction, first.validation_fraction, first.test_fraction) == (
        0.6,
        0.2,
        0.2,
    )
    assert first.algorithm_version == "flickr8k-image-id-split-v1"
    assert len(set(first.train) | set(first.validation) | set(first.test)) == 10
    assert first.provenance[0].sha256 == sha256_file(captions_file)

    output = tmp_path / "splits" / "manifest.json"
    duplicate_output = tmp_path / "manifest-copy.json"
    save_manifest(first, output)
    save_manifest(first, duplicate_output)
    assert output.read_bytes() == duplicate_output.read_bytes()
    loaded = load_manifest(output)
    assert loaded == first
    assert [record.image_id for record in records_for_split(records, loaded, "train")] == list(
        first.train
    )

    captions_file.write_text("tampered caption source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_manifest(first, records, provenance_files={"captions": captions_file})
    captions_file.write_text("stable caption source\n", encoding="utf-8")
    _write_image(records[0].image_path, (255, 255, 255))
    with pytest.raises(ValueError, match="Image-content SHA256"):
        validate_manifest(first, records, provenance_files={"captions": captions_file})


def test_official_manifest_preserves_order_and_records_unassigned_images(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path, 5)
    captions_file = tmp_path / "captions.txt"
    captions_file.write_text("caption bytes\n", encoding="utf-8")
    train_file = tmp_path / "train.txt"
    validation_file = tmp_path / "dev.txt"
    test_file = tmp_path / "test.txt"
    train_file.write_text("image-2.jpg\nimage-0.jpg\n", encoding="utf-8")
    validation_file.write_text("image-1.jpg\n", encoding="utf-8")
    test_file.write_text("image-3.jpg\n", encoding="utf-8")

    manifest = create_official_manifest(
        records,
        captions_file=captions_file,
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        expected_references_per_image=2,
    )

    assert manifest.train == ("image-2.jpg", "image-0.jpg")
    assert manifest.validation == ("image-1.jpg",)
    assert manifest.test == ("image-3.jpg",)
    assert manifest.excluded == ("image-4.jpg",)

    validation_file.write_text("image-2.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        create_official_manifest(
            records,
            captions_file=captions_file,
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            expected_references_per_image=2,
        )


def test_split_and_record_duplicate_validation(tmp_path: Path) -> None:
    duplicate_file = tmp_path / "duplicate.txt"
    duplicate_file.write_text("image.jpg\nimage.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate image ID"):
        read_official_split(duplicate_file)

    records = _records(tmp_path, 3)
    with pytest.raises(ValueError, match="Duplicate image ID"):
        validate_records((records[0], records[0]))

    captions_file = tmp_path / "captions.txt"
    captions_file.write_text("source\n", encoding="utf-8")
    manifest = create_seeded_manifest(
        records,
        captions_file=captions_file,
        seed=2,
        train_fraction=1 / 3,
        validation_fraction=1 / 3,
        test_fraction=1 / 3,
        expected_references_per_image=2,
    )
    payload = copy.deepcopy(manifest.to_dict())
    payload["splits"]["train"].append(payload["splits"]["train"][0])
    with pytest.raises(ValueError, match="duplicate image IDs"):
        SplitManifest.from_dict(payload)


def test_reference_count_is_validated_before_manifest_creation(tmp_path: Path) -> None:
    records = (
        CaptionRecord("one.jpg", tmp_path / "one.jpg", ("only one reference",)),
        CaptionRecord("two.jpg", tmp_path / "two.jpg", ("first", "second")),
    )
    captions_file = tmp_path / "captions.txt"
    captions_file.write_text("source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2"):
        create_seeded_manifest(
            records,
            captions_file=captions_file,
            seed=1,
            expected_references_per_image=2,
        )
