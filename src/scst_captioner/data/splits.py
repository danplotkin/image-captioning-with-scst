"""Deterministic, provenance-checked image-level split manifests."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scst_captioner.data.records import (
    CaptionRecord,
    normalize_image_id,
    validate_records,
)

MANIFEST_SCHEMA_VERSION = 2
SplitName = Literal["train", "validation", "test", "excluded"]


@dataclass(frozen=True, slots=True)
class FileProvenance:
    """A portable source label plus its content digest."""

    role: str
    filename: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("provenance role cannot be empty")
        if not self.filename:
            raise ValueError("provenance filename cannot be empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError(f"Invalid SHA256 digest for provenance role {self.role!r}")

    @classmethod
    def from_path(cls, role: str, path: str | Path) -> FileProvenance:
        source = Path(path)
        return cls(role=role, filename=source.name, sha256=sha256_file(source))


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Ordered image IDs and immutable provenance for one dataset partition."""

    strategy: Literal["official", "seeded"]
    records_sha256: str
    images_sha256: str
    expected_references_per_image: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    provenance: tuple[FileProvenance, ...] = ()
    seed: int | None = None
    train_fraction: float | None = None
    validation_fraction: float | None = None
    test_fraction: float | None = None
    algorithm_version: str = "flickr8k-image-id-split-v1"
    caption_policy: str = "preserve-nonempty-references-v1"
    schema_version: int = MANIFEST_SCHEMA_VERSION
    dataset: str = "flickr8k"

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported split manifest schema {self.schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        if self.strategy not in {"official", "seeded"}:
            raise ValueError(f"Unknown split strategy: {self.strategy!r}")
        if self.strategy == "seeded" and self.seed is None:
            raise ValueError("seeded split manifests must record their seed")
        if self.strategy == "official" and self.seed is not None:
            raise ValueError("official split manifests cannot have a seed")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if self.strategy == "seeded":
            if any(value is None for value in fractions):
                raise ValueError("seeded split manifests must record all split fractions")
            _validate_fractions(*fractions)  # type: ignore[arg-type]
        elif any(value is not None for value in fractions):
            raise ValueError("official split manifests cannot record random split fractions")
        if not self.algorithm_version or not self.caption_policy:
            raise ValueError("manifest algorithm version and caption policy cannot be empty")
        if self.expected_references_per_image < 1:
            raise ValueError("expected_references_per_image must be positive")
        if len(self.records_sha256) != 64:
            raise ValueError("records_sha256 must be a SHA256 hex digest")
        if len(self.images_sha256) != 64:
            raise ValueError("images_sha256 must be a SHA256 hex digest")

        for name in ("train", "validation", "test", "excluded"):
            values = tuple(normalize_image_id(item) for item in getattr(self, name))
            object.__setattr__(self, name, values)
        object.__setattr__(self, "provenance", tuple(self.provenance))
        validate_manifest_structure(self)

    def ids(self, split: SplitName) -> tuple[str, ...]:
        if split not in {"train", "validation", "test", "excluded"}:
            raise ValueError(f"Unknown split name: {split!r}")
        return getattr(self, split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "expected_references_per_image": self.expected_references_per_image,
            "provenance": [
                {
                    "filename": item.filename,
                    "role": item.role,
                    "sha256": item.sha256,
                }
                for item in self.provenance
            ],
            "records_sha256": self.records_sha256,
            "images_sha256": self.images_sha256,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "strategy_parameters": {
                "algorithm_version": self.algorithm_version,
                "caption_policy": self.caption_policy,
                "test_fraction": self.test_fraction,
                "train_fraction": self.train_fraction,
                "validation_fraction": self.validation_fraction,
            },
            "splits": {
                "excluded": list(self.excluded),
                "test": list(self.test),
                "train": list(self.train),
                "validation": list(self.validation),
            },
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SplitManifest:
        expected_keys = {
            "dataset",
            "expected_references_per_image",
            "provenance",
            "records_sha256",
            "images_sha256",
            "schema_version",
            "seed",
            "strategy_parameters",
            "splits",
            "strategy",
        }
        unknown = set(payload) - expected_keys
        missing = expected_keys - set(payload)
        if unknown or missing:
            raise ValueError(
                f"Invalid manifest keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        splits = payload["splits"]
        if not isinstance(splits, Mapping):
            raise ValueError("manifest 'splits' must be an object")
        expected_splits = {"train", "validation", "test", "excluded"}
        if set(splits) != expected_splits:
            raise ValueError("manifest must define train, validation, test, and excluded")

        provenance_payload = payload["provenance"]
        if not isinstance(provenance_payload, list):
            raise ValueError("manifest 'provenance' must be a list")
        provenance: list[FileProvenance] = []
        for item in provenance_payload:
            if not isinstance(item, Mapping) or set(item) != {"role", "filename", "sha256"}:
                raise ValueError("invalid provenance entry")
            provenance.append(
                FileProvenance(
                    role=str(item["role"]),
                    filename=str(item["filename"]),
                    sha256=str(item["sha256"]),
                )
            )

        def _split(name: str) -> tuple[str, ...]:
            value = splits[name]
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"manifest split {name!r} must be a list of strings")
            return tuple(value)

        strategy = payload["strategy"]
        if strategy not in {"official", "seeded"}:
            raise ValueError(f"Unknown split strategy: {strategy!r}")
        seed = payload["seed"]
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("manifest seed must be an integer or null")
        expected_references = payload["expected_references_per_image"]
        if not isinstance(expected_references, int) or isinstance(expected_references, bool):
            raise ValueError("expected_references_per_image must be an integer")
        schema_version = payload["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ValueError("schema_version must be an integer")
        parameters = payload["strategy_parameters"]
        expected_parameters = {
            "algorithm_version",
            "caption_policy",
            "test_fraction",
            "train_fraction",
            "validation_fraction",
        }
        if not isinstance(parameters, Mapping) or set(parameters) != expected_parameters:
            raise ValueError("manifest has invalid strategy_parameters")
        for fraction_name in (
            "train_fraction",
            "validation_fraction",
            "test_fraction",
        ):
            fraction = parameters[fraction_name]
            if fraction is not None and (
                isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            ):
                raise ValueError(f"manifest {fraction_name} must be numeric or null")
        if not isinstance(parameters["algorithm_version"], str) or not isinstance(
            parameters["caption_policy"], str
        ):
            raise ValueError("manifest algorithm_version/caption_policy must be strings")

        return cls(
            dataset=str(payload["dataset"]),
            strategy=strategy,
            seed=seed,
            records_sha256=str(payload["records_sha256"]),
            images_sha256=str(payload["images_sha256"]),
            expected_references_per_image=expected_references,
            train=_split("train"),
            validation=_split("validation"),
            test=_split("test"),
            excluded=_split("excluded"),
            provenance=tuple(provenance),
            schema_version=schema_version,
            algorithm_version=str(parameters["algorithm_version"]),
            caption_policy=str(parameters["caption_policy"]),
            train_fraction=parameters["train_fraction"],
            validation_fraction=parameters["validation_fraction"],
            test_fraction=parameters["test_fraction"],
        )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source file without loading it all into memory."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Provenance source does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records: Sequence[CaptionRecord]) -> str:
    """Hash parsed semantic content independently of machine-local paths."""

    records = validate_records(records)
    canonical = [
        {"captions": list(record.captions), "image_id": record.image_id}
        for record in sorted(records, key=lambda item: item.image_id)
    ]
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_files_sha256(records: Sequence[CaptionRecord]) -> str:
    """Hash ordered image IDs and exact bytes for pixel-level dataset identity."""

    records = validate_records(records)
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.image_id):
        if not record.image_path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint missing image: {record.image_path}")
        digest.update(record.image_id.encode("utf-8"))
        digest.update(b"\0")
        with record.image_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def read_official_split(path: str | Path) -> tuple[str, ...]:
    """Read an official one-image-per-line split while preserving its order."""

    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Official split file does not exist: {split_path}")
    image_ids: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        split_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw_line.strip()
        if not value:
            continue
        image_id = normalize_image_id(value)
        if image_id in seen:
            raise ValueError(
                f"Duplicate image ID {image_id!r} in {split_path} on line {line_number}"
            )
        seen.add(image_id)
        image_ids.append(image_id)
    if not image_ids:
        raise ValueError(f"Official split file is empty: {split_path}")
    return tuple(image_ids)


def validate_manifest_structure(manifest: SplitManifest) -> None:
    """Reject duplicates, overlaps, or duplicate provenance roles."""

    split_sets: dict[str, set[str]] = {}
    for name in ("train", "validation", "test", "excluded"):
        values = manifest.ids(name)
        if len(values) != len(set(values)):
            raise ValueError(f"Split {name!r} contains duplicate image IDs")
        split_sets[name] = set(values)
    for index, left in enumerate(split_sets):
        for right in tuple(split_sets)[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:3])
                raise ValueError(f"Splits {left!r} and {right!r} overlap: {examples}")
    roles = [item.role for item in manifest.provenance]
    if len(roles) != len(set(roles)):
        raise ValueError("manifest contains duplicate provenance roles")


def validate_manifest(
    manifest: SplitManifest,
    records: Sequence[CaptionRecord],
    *,
    provenance_files: Mapping[str, str | Path] | None = None,
) -> None:
    """Validate IDs, references, semantic digest, and optional source files."""

    records = validate_records(
        records,
        expected_references_per_image=manifest.expected_references_per_image,
    )
    validate_manifest_structure(manifest)
    record_ids = {record.image_id for record in records}
    manifest_ids = (
        set(manifest.train) | set(manifest.validation) | set(manifest.test) | set(manifest.excluded)
    )
    unknown = manifest_ids - record_ids
    missing = record_ids - manifest_ids
    if unknown:
        raise ValueError(f"Manifest contains unknown image IDs: {', '.join(sorted(unknown)[:3])}")
    if missing:
        raise ValueError(f"Manifest omits image IDs: {', '.join(sorted(missing)[:3])}")
    current_records_sha = records_sha256(records)
    if current_records_sha != manifest.records_sha256:
        raise ValueError(
            "Caption-record SHA256 does not match the split manifest; dataset content changed"
        )
    if image_files_sha256(records) != manifest.images_sha256:
        raise ValueError("Image-content SHA256 does not match the split manifest")

    if provenance_files is not None:
        recorded = {entry.role: entry for entry in manifest.provenance}
        unknown_roles = set(provenance_files) - set(recorded)
        if unknown_roles:
            raise ValueError(
                f"No recorded provenance for roles: {', '.join(sorted(unknown_roles))}"
            )
        for role, path in provenance_files.items():
            actual = sha256_file(path)
            if actual != recorded[role].sha256:
                raise ValueError(f"SHA256 mismatch for provenance source {role!r}")


def _validate_fractions(train: float, validation: float, test: float) -> None:
    fractions = (train, validation, test)
    if any(not math.isfinite(value) or value < 0 for value in fractions):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train, validation, and test fractions must sum to one")


def create_seeded_manifest(
    records: Sequence[CaptionRecord],
    *,
    captions_file: str | Path,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    expected_references_per_image: int = 5,
    caption_policy: str = "preserve-nonempty-references-v1",
) -> SplitManifest:
    """Create a deterministic custom split from a sorted image-ID population."""

    records = validate_records(
        records,
        expected_references_per_image=expected_references_per_image,
    )
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    image_ids = sorted(record.image_id for record in records)
    random.Random(seed).shuffle(image_ids)
    train_end = int(len(image_ids) * train_fraction)
    validation_end = train_end + int(len(image_ids) * validation_fraction)
    manifest = SplitManifest(
        strategy="seeded",
        seed=seed,
        records_sha256=records_sha256(records),
        images_sha256=image_files_sha256(records),
        expected_references_per_image=expected_references_per_image,
        train=tuple(image_ids[:train_end]),
        validation=tuple(image_ids[train_end:validation_end]),
        test=tuple(image_ids[validation_end:]),
        provenance=(FileProvenance.from_path("captions", captions_file),),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        caption_policy=caption_policy,
    )
    validate_manifest(manifest, records, provenance_files={"captions": captions_file})
    return manifest


def create_official_manifest(
    records: Sequence[CaptionRecord],
    *,
    captions_file: str | Path,
    train_file: str | Path,
    validation_file: str | Path,
    test_file: str | Path,
    expected_references_per_image: int = 5,
    caption_policy: str = "preserve-nonempty-references-v1",
) -> SplitManifest:
    """Create a manifest from Flickr8K's official train/dev/test text files.

    The historical official files do not necessarily assign every Flickr8K
    image.  Such images are recorded explicitly in ``excluded`` rather than
    disappearing from provenance.
    """

    records = validate_records(
        records,
        expected_references_per_image=expected_references_per_image,
    )
    train = read_official_split(train_file)
    validation = read_official_split(validation_file)
    test = read_official_split(test_file)
    assigned = set(train) | set(validation) | set(test)
    known = {record.image_id for record in records}
    unknown = assigned - known
    if unknown:
        raise ValueError(
            f"Official split files contain unknown image IDs: {', '.join(sorted(unknown)[:3])}"
        )
    excluded = tuple(record.image_id for record in records if record.image_id not in assigned)
    sources = {
        "captions": captions_file,
        "official_test": test_file,
        "official_train": train_file,
        "official_validation": validation_file,
    }
    manifest = SplitManifest(
        strategy="official",
        records_sha256=records_sha256(records),
        images_sha256=image_files_sha256(records),
        expected_references_per_image=expected_references_per_image,
        train=train,
        validation=validation,
        test=test,
        excluded=excluded,
        provenance=tuple(
            FileProvenance.from_path(role, source) for role, source in sources.items()
        ),
        caption_policy=caption_policy,
    )
    validate_manifest(manifest, records, provenance_files=sources)
    return manifest


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    """Write canonical JSON while preserving the order of every ID list."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output.write_text(f"{serialized}\n", encoding="utf-8")


def load_manifest(path: str | Path) -> SplitManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid split-manifest JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("split manifest root must be a JSON object")
    return SplitManifest.from_dict(payload)


def records_for_split(
    records: Sequence[CaptionRecord],
    manifest: SplitManifest,
    split: Literal["train", "validation", "test"],
    *,
    validate: bool = True,
) -> tuple[CaptionRecord, ...]:
    """Select records in manifest order after validating the full manifest."""

    if validate:
        validate_manifest(manifest, records)
    by_id = {record.image_id: record for record in records}
    return tuple(by_id[image_id] for image_id in manifest.ids(split))
