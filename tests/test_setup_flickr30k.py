from __future__ import annotations

import hashlib
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from scst_captioner.cli import _load_corpus
from scst_captioner.config import load_config
from scst_captioner.data import load_flickr8k_records, load_manifest


@pytest.fixture
def setup_script():
    path = Path(__file__).resolve().parents[1] / "scripts/setup_flickr30k.py"
    spec = importlib.util.spec_from_file_location("setup_flickr30k", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus(setup_script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(setup_script, "SPLIT_COUNTS", {"train": 1, "val": 1, "test": 1})
    config = Path(__file__).resolve().parents[1] / "configs/flickr30k.toml"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/flickr30k.toml").write_bytes(config.read_bytes())
    directory = tmp_path / "data/flickr30k"
    images = directory / "flickr30k_images/flickr30k_images"
    images.mkdir(parents=True)
    annotations = {
        "dataset": "flickr30k",
        "images": [
            {
                "filename": f"{index}.jpg",
                "split": split,
                "sentences": [{"raw": 'A child says, "hello".'}] * 5,
            }
            for index, split in enumerate(("train", "val", "test"))
        ],
    }
    source = directory / setup_script.ANNOTATION_NAME
    source.write_text(json.dumps(annotations))
    monkeypatch.setattr(setup_script, "ANNOTATION_HASH", hashlib.sha256(source.read_bytes()).hexdigest())
    for index in range(3):
        Image.new("RGB", (3, 3), (index, index, index)).save(images / f"{index}.jpg")
    return directory, source, annotations


def _no_network(*args, **kwargs):
    pytest.fail("Verified local assets should not trigger a download")


def test_setup_is_repeatable_and_loads_through_training_cli(
    setup_script, corpus, monkeypatch
):
    directory, _, _ = corpus
    monkeypatch.setattr(setup_script, "urlopen", _no_network)
    setup_script.main()
    config = load_config("configs/flickr30k.toml")
    records, manifest = _load_corpus(config)
    assert manifest.dataset == "flickr30k"
    assert manifest.algorithm_version == "flickr30k-karpathy-split-v1"
    assert (manifest.train, manifest.validation, manifest.test, manifest.excluded) == (
        ("0.jpg",), ("1.jpg",), ("2.jpg",), ()
    )
    assert records[0].captions == ('A child says, "hello".',) * 5
    paths = [directory / "captions.csv", Path(config.data.split_manifest)]
    originals = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    setup_script.main()
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == originals
    Image.new("RGB", (3, 3), "red").save(records[0].image_path)
    with pytest.raises(ValueError, match="Existing manifest differs"):
        setup_script.main()
    assert paths[1].read_bytes() == originals[1][0]


def test_missing_images_trigger_pinned_kaggle_download(setup_script, corpus, monkeypatch):
    import sys

    directory, _, _ = corpus
    image = directory / "flickr30k_images/flickr30k_images/0.jpg"
    payload = image.read_bytes()
    image.unlink()
    calls = []

    def download(dataset, *, output_dir):
        calls.append((dataset, output_dir))
        image.write_bytes(payload)

    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=download))
    setup_script.main()
    assert calls == [(setup_script.DATASET, "data/flickr30k")]
    assert load_manifest("data/splits/flickr30k.json").train == ("0.jpg",)


def test_annotations_verified_before_writing(setup_script, tmp_path, monkeypatch):
    payload = b'{"dataset": "flickr30k"}'
    monkeypatch.setattr(setup_script, "ANNOTATION_HASH", hashlib.sha256(payload).hexdigest())

    def response(content):
        stream = BytesIO()
        with ZipFile(stream, "w") as archive:
            archive.writestr(setup_script.ANNOTATION_NAME, content)
            archive.writestr("../unwanted.txt", b"ignored")
        stream.seek(0)
        return stream

    monkeypatch.setattr(setup_script, "urlopen", lambda *a, **kw: response(b"changed"))
    with pytest.raises(ValueError, match="failed SHA256"):
        setup_script.ensure_annotations(tmp_path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(setup_script, "urlopen", lambda *a, **kw: response(payload))
    source = setup_script.ensure_annotations(tmp_path)
    assert source.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [source]
    monkeypatch.setattr(setup_script, "urlopen", _no_network)
    assert setup_script.ensure_annotations(tmp_path) == source
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differ from the pinned source"):
        setup_script.ensure_annotations(tmp_path)
    assert source.read_bytes() == b"changed"


@pytest.mark.parametrize("change", ["duplicate", "unknown_split", "references", "counts"])
def test_invalid_annotations_do_not_write_captions(setup_script, corpus, change):
    directory, source, annotations = corpus
    if change == "duplicate":
        annotations["images"][1]["filename"] = "0.jpg"
    elif change == "unknown_split":
        annotations["images"][0]["split"] = "restval"
    elif change == "references":
        annotations["images"][0]["sentences"].pop()
    else:
        annotations["images"].pop()
    source.write_text(json.dumps(annotations))
    with pytest.raises(ValueError):
        setup_script.prepare_captions(source, directory / "captions.csv")
    assert not (directory / "captions.csv").exists()


def test_changed_captions_rejected_without_overwrite(setup_script, corpus):
    directory, source, _ = corpus
    output = directory / "captions.csv"
    output.write_text("changed")
    with pytest.raises(ValueError, match="Existing captions differ"):
        setup_script.prepare_captions(source, output)
    assert output.read_text() == "changed"


def test_missing_image_fails_validation(setup_script, corpus):
    directory, source, _ = corpus
    captions = directory / "captions.csv"
    setup_script.prepare_captions(source, captions)
    images = directory / "flickr30k_images/flickr30k_images"
    (images / "0.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_flickr8k_records(captions, images)
