import pickle
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.torch_version import TorchVersion

from scst_captioner.checkpointing import (
    build_checkpoint,
    load_checkpoint_file,
    load_model_checkpoint,
    restore_training_state,
    save_checkpoint,
    sha256_file,
)
from scst_captioner.reproducibility import collect_environment


class UnsupportedMetadata:
    pass


def test_versioned_checkpoint_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    checkpoint = build_checkpoint(
        model,
        stage="xe",
        config={"model": {"d_model": 3}},
        include_rng_state=False,
    )
    path = tmp_path / "model.pt"
    save_checkpoint(checkpoint, path)
    clone = nn.Linear(3, 2)
    loaded = load_model_checkpoint(clone, path)
    assert loaded["schema_version"] == 1
    assert sha256_file(path)
    for expected, actual in zip(model.parameters(), clone.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_raw_legacy_state_dict_loads_strictly(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), path)
    clone = nn.Linear(2, 2)
    metadata = load_model_checkpoint(clone, path)
    assert metadata["legacy"] is True
    assert metadata["provenance"]["lineage"] == "unknown"


def test_checkpoint_with_rng_state_uses_safe_loader(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    path = tmp_path / "rng.pt"
    save_checkpoint(
        build_checkpoint(model, stage="xe", config={}, include_rng_state=True),
        path,
    )
    loaded = load_model_checkpoint(nn.Linear(1, 1), path)
    assert "rng_state" in loaded


def test_checkpoint_with_real_environment_loads_with_weights_only(tmp_path: Path) -> None:
    environment = collect_environment()
    assert type(environment["torch"]) is str
    model = nn.Linear(2, 1)
    path = tmp_path / "production-provenance.pt"
    save_checkpoint(
        build_checkpoint(model, stage="xe", config={}, provenance={"environment": environment}),
        path,
    )
    # No allowlist is needed for newly written production metadata.
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["provenance"]["environment"] == environment
    clone = nn.Linear(2, 1)
    load_model_checkpoint(clone, path)
    assert torch.equal(clone.weight, model.weight)


def test_older_torch_version_metadata_loads_without_global_allowlist(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    path = tmp_path / "older-provenance.pt"
    save_checkpoint(
        build_checkpoint(
            model,
            stage="xe",
            config={},
            provenance={"environment": {"torch": TorchVersion("2.4.0")}},
        ),
        path,
    )
    allowed_before = set(torch.serialization.get_safe_globals())
    loaded = load_model_checkpoint(nn.Linear(1, 1), path)
    assert loaded["provenance"]["environment"]["torch"] == "2.4.0"
    assert set(torch.serialization.get_safe_globals()) == allowed_before


def test_checkpoint_loader_still_rejects_unrelated_objects(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.pt"
    torch.save({"metadata": UnsupportedMetadata()}, path)
    with pytest.raises(pickle.UnpicklingError, match="UnsupportedMetadata"):
        load_checkpoint_file(path)


def test_checkpoint_loader_preserves_callers_torch_version_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "nested-safe-globals.pt"
    torch.save({"torch": TorchVersion("2.4.0")}, path)
    with torch.serialization.safe_globals([TorchVersion]):
        allowed_before = set(torch.serialization.get_safe_globals())
        assert load_checkpoint_file(path)["torch"] == "2.4.0"
        assert set(torch.serialization.get_safe_globals()) == allowed_before
        assert torch.load(path, weights_only=True)["torch"] == "2.4.0"


def test_full_optimizer_scheduler_scaler_state_restores_with_safe_loader(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "full.pt"
    save_checkpoint(
        build_checkpoint(
            model,
            stage="xe",
            config={},
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        ),
        path,
    )

    clone = nn.Linear(2, 1)
    clone_optimizer = torch.optim.Adam(clone.parameters(), lr=1.0)
    clone_scheduler = torch.optim.lr_scheduler.StepLR(clone_optimizer, step_size=1)
    clone_scaler = torch.amp.GradScaler("cuda", enabled=False)
    loaded = load_model_checkpoint(clone, path)
    restore_training_state(
        loaded,
        optimizer=clone_optimizer,
        scheduler=clone_scheduler,
        scaler=clone_scaler,
        strict=True,
    )
    assert clone_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert clone_scheduler.last_epoch == scheduler.last_epoch
