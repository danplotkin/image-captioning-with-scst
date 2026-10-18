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

    no_baseline = dataclasses.replace(
        config, scst=dataclasses.replace(config.scst, baseline="none")
    )
    with pytest.raises(ValueError, match="reward/baseline/sampling"):
        cli._validate_checkpoint_scientific_protocol(checkpoint, no_baseline)

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


def _resumable_checkpoint(tmp_path: Path, stage: str):
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n", encoding="utf-8")
    config = load_config("configs/default.toml")
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, split_manifest=str(split_path))
    )
    manifest = SimpleNamespace(records_sha256="a" * 64, images_sha256="b" * 64)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    provenance = {
        "split_manifest_sha256": sha256_file(split_path),
        "caption_records_sha256": manifest.records_sha256,
        "image_files_sha256": manifest.images_sha256,
    }
    if stage == "scst":
        provenance["reward_protocol"] = cli._reward_protocol_metadata(config)
    checkpoint = build_checkpoint(
        model,
        stage=stage,
        config=config.to_dict(),
        epoch=3,
        global_step=7,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    checkpoint["provenance"] = provenance
    return config, manifest, checkpoint


@pytest.mark.parametrize("stage, extra_epochs", [("xe", 0), ("scst", 0), ("scst", 15)])
def test_resume_accepts_current_protocol_and_restores_training_state(
    tmp_path: Path, stage: str, extra_epochs: int
) -> None:
    config, manifest, checkpoint = _resumable_checkpoint(tmp_path, stage)
    path = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, path)
    if extra_epochs:
        config = dataclasses.replace(
            config, scst=dataclasses.replace(config.scst, epochs=config.scst.epochs + extra_epochs)
        )
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch, global_step, summary = cli._resume_if_requested(
        str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        manifest=manifest,
        expected_stage=stage,
    )
    assert (start_epoch, global_step) == (4, 7)
    assert summary is not None and summary["stage"] == stage
    assert optimizer.state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize("change", [{"baseline": "none"}, {"learning_rate": 0.1}, {"epochs": 1}])
@pytest.mark.parametrize("schedule", ["exponential", "constant"])
def test_rl_epoch_extension_keeps_resume_settings_strict(
    tmp_path: Path, change: dict, schedule: str
) -> None:
    config, manifest, checkpoint = _resumable_checkpoint(tmp_path, "scst")
    path = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, path)
    changes = {"epochs": config.scst.epochs + 15, "lr_schedule": schedule, **change}
    config = dataclasses.replace(config, scst=dataclasses.replace(config.scst, **changes))
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with pytest.raises(ValueError, match="configuration does not match"):
        cli._resume_if_requested(
            str(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            config=config,
            manifest=manifest,
            expected_stage="scst",
        )
    assert not optimizer.state


@pytest.mark.parametrize(
    "changed_field", ["document_frequency", "tokenization", "package_version", "missing_protocol"]
)
def test_scst_resume_rejects_obsolete_reward_before_restoring_training_state(
    tmp_path: Path, changed_field: str
) -> None:
    config, manifest, checkpoint = _resumable_checkpoint(tmp_path, "scst")
    if changed_field == "missing_protocol":
        del checkpoint["provenance"]["reward_protocol"]
    else:
        checkpoint["provenance"]["reward_protocol"][changed_field] = "obsolete"
    path = tmp_path / "obsolete.pt"
    save_checkpoint(checkpoint, path)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="reward implementation"):
        cli._resume_if_requested(
            str(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            manifest=manifest,
            expected_stage="scst",
        )
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert scheduler.last_epoch == 0
    assert torch.equal(torch.get_rng_state(), rng_before)


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


@pytest.mark.parametrize("legacy_config", [False, True])
@pytest.mark.parametrize("override_lr", [False, True])
def test_constant_resume_preserves_adam_and_checkpoint_lrs(
    tmp_path: Path, legacy_config: bool, override_lr: bool
) -> None:
    config, manifest, checkpoint = _resumable_checkpoint(tmp_path, "scst")
    if legacy_config:
        checkpoint["config"]["scst"].pop("lr_schedule")
        checkpoint["config"]["scst"].pop("resume_use_config_lrs")
    path = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, path)
    config = dataclasses.replace(
        config,
        scst=dataclasses.replace(
            config.scst,
            epochs=30,
            lr_schedule="constant",
            learning_rate=2e-6 if override_lr else config.scst.learning_rate,
            resume_use_config_lrs=override_lr,
        ),
    )
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.scst.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start, _, _ = cli._resume_if_requested(
        str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        manifest=manifest,
        expected_stage="scst",
    )
    assert start == 4
    expected_lr = (
        config.scst.learning_rate
        if override_lr
        else checkpoint["optimizer_state"]["param_groups"][0]["lr"]
    )
    assert optimizer.param_groups[0]["lr"] == expected_lr
    assert scheduler.last_epoch == checkpoint["scheduler_state"]["last_epoch"]
    assert scheduler.gamma == 1.0
    assert scheduler.get_last_lr() == [expected_lr]
    for key, state in checkpoint["optimizer_state"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][name], value)
    for _ in range(15):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == expected_lr
    # A subsequent resume must also keep the constant schedule and counters.
    saved = build_checkpoint(
        model,
        stage="scst",
        config=config.to_dict(),
        epoch=18,
        global_step=22,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    saved["provenance"] = checkpoint["provenance"]
    save_checkpoint(saved, path)
    scheduler.gamma = 0.85
    start, _, _ = cli._resume_if_requested(
        str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        manifest=manifest,
        expected_stage="scst",
    )
    assert start == 19
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == expected_lr
    assert scheduler.gamma == 1.0


def test_old_adam_checkpoint_allows_inference_but_rejects_adamw_resume(tmp_path):
    config, manifest, checkpoint = _resumable_checkpoint(tmp_path, "scst")
    del checkpoint["config"]["scst"]["weight_decay"]
    config = dataclasses.replace(config, scst=dataclasses.replace(config.scst, weight_decay=0.01))
    cli._validate_checkpoint_scientific_protocol(checkpoint, config)
    path = tmp_path / "old-adam.pt"
    save_checkpoint(checkpoint, path)
    model = nn.Linear(2, 2)
    optimizer = cli.build_scst_optimizer(model, config.scst)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    with pytest.raises(ValueError, match="predates AdamW"):
        cli._resume_if_requested(
            str(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            config=config,
            manifest=manifest,
            expected_stage="scst",
        )


def test_resume_rejects_weight_decay_change(tmp_path):
    config, manifest, checkpoint = _resumable_checkpoint(tmp_path, "scst")
    config = dataclasses.replace(config, scst=dataclasses.replace(config.scst, weight_decay=0.01))
    path = tmp_path / "different-decay.pt"
    save_checkpoint(checkpoint, path)
    model = nn.Linear(2, 2)
    optimizer = cli.build_scst_optimizer(model, config.scst)
    with pytest.raises(ValueError, match="configuration"):
        cli._resume_if_requested(
            str(path),
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            config=config,
            manifest=manifest,
            expected_stage="scst",
        )
