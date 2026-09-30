from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from scst_captioner import cli
from scst_captioner.checkpointing import build_checkpoint, save_checkpoint, sha256_file
from scst_captioner.config import load_config
from scst_captioner.evaluation import PredictionRecord


class NumberDataset(torch.utils.data.Dataset[int]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return index


def test_loader_generator_state_reproduces_next_epoch_order() -> None:
    config = load_config("configs/default.toml")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, num_workers=0),
    )
    uninterrupted = cli._loader(NumberDataset(), batch_size=2, shuffle=True, config=config)
    list(uninterrupted)
    assert uninterrupted.generator is not None
    saved_state = uninterrupted.generator.get_state()
    expected = [batch.tolist() for batch in uninterrupted]

    resumed = cli._loader(NumberDataset(), batch_size=2, shuffle=True, config=config)
    cli._restore_loader_generator(resumed, saved_state)
    actual = [batch.tolist() for batch in resumed]
    assert actual == expected


def test_checkpoint_protocol_rejects_scst_reward_mismatch_and_raw_stage() -> None:
    config = load_config("configs/default.toml")
    checkpoint = {
        "stage": "scst",
        "config": config.to_dict(),
        "provenance": {"reward_protocol": cli._reward_protocol_metadata(config)},
    }
    cli._validate_checkpoint_scientific_protocol(checkpoint, config)

    changed = json.loads(json.dumps(checkpoint))
    changed["config"]["scst"]["sample_model_mode"] = "train"
    with pytest.raises(ValueError, match="reward/baseline/sampling"):
        cli._validate_checkpoint_scientific_protocol(changed, config)

    old_cider = json.loads(json.dumps(checkpoint))
    old_cider["provenance"]["reward_protocol"] = {
        "name": "cider_d",
        "implementation": "pycocoevalcap.cider.cider_scorer.CiderScorer",
        "package_version": "1.2",
        "tokenization": "nltk_treebank_lowercase_coco_punctuation_v1",
        "eos_policy": "append_on_completion",
    }
    with pytest.raises(ValueError, match="reward implementation"):
        cli._validate_checkpoint_scientific_protocol(old_cider, config)

    with pytest.raises(ValueError, match="unverified stage"):
        cli._validate_checkpoint_config({"legacy": True}, config, expected_stage="xe")
    cli._validate_checkpoint_config(
        {"legacy": True}, config, expected_stage="xe", allow_legacy=True
    )


def test_nonresumable_envelope_fails_before_training(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n", encoding="utf-8")
    config = load_config("configs/default.toml")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, split_manifest=str(split_path)),
    )
    manifest = SimpleNamespace(records_sha256="a" * 64, images_sha256="c" * 64)
    model = nn.Linear(2, 2)
    path = tmp_path / "not-resumable.pt"
    save_checkpoint(
        build_checkpoint(
            model,
            stage="xe",
            config=config.to_dict(),
            provenance={
                "split_manifest_sha256": sha256_file(split_path),
                "caption_records_sha256": manifest.records_sha256,
                "image_files_sha256": manifest.images_sha256,
            },
            include_rng_state=False,
        ),
        path,
    )
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(ValueError, match="cannot resume training"):
        cli._resume_if_requested(
            str(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            manifest=manifest,
            expected_stage="xe",
        )


def test_prediction_payload_hash_surface_includes_completion() -> None:
    predictions = [PredictionRecord("image.jpg", "a caption")]
    objective = {"per_image": [{"image_id": "image.jpg", "score": 0.5, "completed": False}]}
    assert cli._prediction_payload(predictions, objective) == [
        {"image_id": "image.jpg", "caption": "a caption", "completed": False}
    ]


class FakeResult:
    def __init__(self) -> None:
        self.provenance = {
            "comparison_protocol_sha256": "protocol",
            "ordered_image_ids_sha256": "ids",
            "references_sha256": "refs",
            "backend": "fake-coco",
            "backend_version": "1",
            "metrics": ["Bleu_1", "METEOR", "CIDEr"],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": {"CIDEr": 1.0}, "provenance": self.provenance}


def test_matrix_command_preflights_lineage_and_writes_four_complete_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n", encoding="utf-8")
    xe_path = tmp_path / "xe.pt"
    scst_path = tmp_path / "scst.pt"
    xe_path.write_bytes(b"xe")
    scst_path.write_bytes(b"scst")
    config = load_config("configs/default.toml")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, split_manifest=str(split_path)),
    )
    manifest = SimpleNamespace(records_sha256="b" * 64, images_sha256="d" * 64)
    tokenizer = object()
    train_records = [object()]
    test_records = [SimpleNamespace(image_id="image.jpg")]
    loader = object()
    model = nn.Linear(1, 1)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "_evaluation_setup",
        lambda _: (
            torch.device("cpu"),
            manifest,
            tokenizer,
            train_records,
            test_records,
            loader,
            model,
        ),
    )

    def fake_preflight(path: str, *, expected_stage: str, **kwargs: Any) -> dict[str, Any]:
        del path, kwargs
        provenance = (
            {"xe_checkpoint_sha256": sha256_file(xe_path)} if expected_stage == "scst" else {}
        )
        return {"stage": expected_stage, "provenance": provenance}

    calls: list[tuple[str, str]] = []

    def fake_evaluate(**kwargs: Any):
        calls.append((kwargs["expected_stage"], kwargs["decoding"]))
        prediction = PredictionRecord("image.jpg", "a caption")
        summary = {"stage": kwargs["expected_stage"], "provenance": {}}
        objective = {
            "name": config.scst.reward,
            "mean": 0.5,
            "per_image": [{"image_id": "image.jpg", "score": 0.5, "completed": True}],
        }
        return FakeResult(), [prediction], summary, objective

    monkeypatch.setattr(cli, "_preflight_checkpoint", fake_preflight)
    monkeypatch.setattr(cli, "_evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(cli, "collect_environment", lambda _: {"test": True})
    output_dir = tmp_path / "matrix"
    args = argparse.Namespace(
        config="unused.toml",
        xe_checkpoint=str(xe_path),
        scst_checkpoint=str(scst_path),
        output_dir=str(output_dir),
        overwrite=False,
    )

    assert cli.command_evaluate_matrix(args) == 0
    assert calls == [
        ("xe", "greedy"),
        ("xe", "beam"),
        ("scst", "greedy"),
        ("scst", "beam"),
    ]
    matrix = json.loads((output_dir / "matrix.json").read_text(encoding="utf-8"))
    assert set(matrix["cells"]) == {"xe", "scst"}
    for stage in ("xe", "scst"):
        for decoding in ("greedy", "beam"):
            cell = matrix["cells"][stage][decoding]
            assert cell["training_objective"]["mean"] == 0.5
            artifact = output_dir / cell["prediction_artifact"]["path"]
            assert sha256_file(artifact) == cell["prediction_artifact"]["sha256"]
