"""Versioned checkpoints and safe legacy state-dict loading."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scst_captioner.reproducibility import capture_rng_state

CHECKPOINT_SCHEMA_VERSION = 1


def _cpu_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        result[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    return result


def build_checkpoint(
    model: nn.Module,
    *,
    stage: str,
    config: dict[str, Any],
    epoch: int = 0,
    global_step: int = 0,
    best_metric: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    provenance: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
    include_rng_state: bool = True,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "model_state": _cpu_state_dict(model.state_dict()),
        "config": config,
        "progress": {"epoch": epoch, "global_step": global_step},
        "best_metric": best_metric,
        "provenance": provenance or {},
        "training_state": training_state or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    if include_rng_state:
        checkpoint["rng_state"] = capture_rng_state()
    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Atomically save a checkpoint on the same filesystem as its destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(checkpoint, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint_file(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load tensor-only legacy files or versioned checkpoints with safe unpickling."""

    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only.
        loaded = torch.load(path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise TypeError("checkpoint must contain a mapping")
    return loaded


def is_legacy_state_dict(value: Mapping[str, Any]) -> bool:
    return bool(value) and all(
        isinstance(key, str) and isinstance(item, torch.Tensor) for key, item in value.items()
    )


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load either the notebook's raw state dict or a versioned envelope."""

    loaded = load_checkpoint_file(path, map_location=map_location)
    if is_legacy_state_dict(loaded):
        model.load_state_dict(loaded, strict=strict)
        return {
            "schema_version": 0,
            "stage": "unknown",
            "legacy": True,
            "provenance": {"lineage": "unknown"},
        }

    version = loaded.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema version: {version!r}")
    model_state = loaded.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("versioned checkpoint is missing model_state")
    model.load_state_dict(model_state, strict=strict)
    return loaded


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = False,
) -> None:
    required = {
        key
        for key, value in (
            ("optimizer_state", optimizer),
            ("scheduler_state", scheduler),
            ("scaler_state", scaler),
        )
        if value is not None
    }
    missing = sorted(required - set(checkpoint))
    if strict and missing:
        raise ValueError(f"Checkpoint cannot resume training; missing {', '.join(missing)}")
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
