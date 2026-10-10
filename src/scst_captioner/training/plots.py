"""Export training history without requiring a display or opening plot windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def export_loss_curve(
    history: Sequence[Mapping[str, Any]], run_dir: Path, *, stage: Literal["xe", "scst"]
) -> None:
    """Refresh the loss plot from all completed epochs, including resumed history."""
    if not history:
        return
    figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figure)
    try:
        axes = figure.subplots()
        epochs = [row["epoch"] for row in history]
        axes.plot(epochs, [row["train"]["loss"] for row in history], marker="o", label="Train")
        if stage == "xe":
            axes.plot(
                epochs,
                [row["validation"]["loss"] for row in history],
                marker="o",
                label="Validation",
            )
            axes.set(title="XE loss", ylabel="Cross-entropy per non-padding token")
        else:
            axes.set(title="SCST training loss", ylabel="REINFORCE loss per image")
        axes.set_xlabel("Epoch")
        axes.set_xticks(epochs)
        axes.grid(alpha=0.3)
        axes.legend()
        figure.savefig(run_dir / "loss_curve.png", dpi=150)
    finally:
        figure.clear()
