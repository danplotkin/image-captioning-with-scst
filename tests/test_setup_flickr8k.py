from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest
from PIL import Image

from scst_captioner.data import load_manifest


@pytest.fixture
def setup_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/setup_flickr8k.py"
    spec = importlib.util.spec_from_file_location("setup_flickr8k", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(payloads: dict[str, bytes]) -> BytesIO:
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    stream.seek(0)
    return stream


def _no_network(*args, **kwargs):
    pytest.fail("Existing verified files should not trigger a download")


def test_existing_verified_lists_need_no_network(setup_script, tmp_path, monkeypatch):
    payload = b"one.jpg\n"
    (tmp_path / "train.txt").write_bytes(payload)
    monkeypatch.setattr(
        setup_script, "SPLIT_HASHES", {"train.txt": hashlib.sha256(payload).hexdigest()}
    )
    monkeypatch.setattr(setup_script, "urlopen", _no_network)
    setup_script.ensure_split_lists(tmp_path)


def test_changed_local_list_is_rejected_without_overwriting(setup_script, tmp_path, monkeypatch):
    path = tmp_path / "train.txt"
    path.write_bytes(b"different.jpg\n")
    monkeypatch.setattr(
        setup_script, "SPLIT_HASHES", {"train.txt": hashlib.sha256(b"one.jpg\n").hexdigest()}
    )
    monkeypatch.setattr(setup_script, "urlopen", _no_network)
    with pytest.raises(ValueError, match="differs from the pinned source"):
        setup_script.ensure_split_lists(tmp_path)
    assert path.read_bytes() == b"different.jpg\n"


def test_download_writes_only_verified_missing_lists(setup_script, tmp_path, monkeypatch):
    payloads = {"train.txt": b"one.jpg\n", "test.txt": b"two.jpg\n"}
    monkeypatch.setattr(
        setup_script,
        "SPLIT_HASHES",
        {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
    )
    existing = tmp_path / "train.txt"
    existing.write_bytes(payloads["train.txt"])
    modified = existing.stat().st_mtime_ns
    monkeypatch.setattr(
        setup_script,
        "urlopen",
        lambda *args, **kwargs: _archive({**payloads, "../unwanted.txt": b"ignored"}),
    )
    setup_script.ensure_split_lists(tmp_path)
    assert existing.stat().st_mtime_ns == modified
    assert (tmp_path / "test.txt").read_bytes() == payloads["test.txt"]
    assert {path.name for path in tmp_path.iterdir()} == set(payloads)


def test_changed_download_does_not_write_any_lists(setup_script, tmp_path, monkeypatch):
    payloads = {"train.txt": b"one.jpg\n", "test.txt": b"two.jpg\n"}
    monkeypatch.setattr(
        setup_script,
        "SPLIT_HASHES",
        {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
    )
    monkeypatch.setattr(
        setup_script,
        "urlopen",
        lambda *args, **kwargs: _archive({**payloads, "test.txt": b"changed.jpg\n"}),
    )
    with pytest.raises(ValueError, match="failed SHA256"):
        setup_script.ensure_split_lists(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_setup_creates_repeatable_manifest_and_rejects_changed_images(
    setup_script, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_script, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.toml").write_text("")
    directory = tmp_path / "data/flickr8k"
    images = directory / "Images"
    images.mkdir(parents=True)
    captions = directory / "captions.txt"
    rows = ["image,caption"]
    for index in range(4):
        Image.new("RGB", (3, 3), (index, index, index)).save(images / f"{index}.jpg")
        rows.extend(f"{index}.jpg,A valid repeated caption" for _ in range(5))
    captions.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(
        setup_script, "CAPTIONS_HASH", hashlib.sha256(captions.read_bytes()).hexdigest()
    )
    hashes = {}
    for index, name in enumerate(setup_script.SPLIT_HASHES):
        path = directory / name
        path.write_text(f"{index}.jpg\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(setup_script, "SPLIT_HASHES", hashes)
    monkeypatch.setattr(setup_script, "urlopen", _no_network)
    setup_script.main()
    output = tmp_path / "data/splits/flickr8k.json"
    manifest = load_manifest(output)
    assert (manifest.train, manifest.validation, manifest.test, manifest.excluded) == (
        ("0.jpg",),
        ("1.jpg",),
        ("2.jpg",),
        ("3.jpg",),
    )
    original = output.read_bytes()
    modified = output.stat().st_mtime_ns
    setup_script.main()
    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == modified
    Image.new("RGB", (3, 3), "red").save(images / "0.jpg")
    with pytest.raises(ValueError, match="Existing manifest differs"):
        setup_script.main()
    assert output.read_bytes() == original
