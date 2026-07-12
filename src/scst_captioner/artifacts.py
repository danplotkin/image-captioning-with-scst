"""Small helpers for auditable run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_run_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    directory = Path(path)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory is not empty: {directory}. Pass --overwrite or choose another path."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory
