from pathlib import Path

import pytest
from matplotlib.figure import Figure

from scst_captioner.training.plots import export_loss_curve


@pytest.mark.parametrize("stage", ["xe", "scst"])
def test_loss_curve_uses_recorded_epochs_and_losses(tmp_path, monkeypatch, stage):
    history = [
        {
            "epoch": 1,
            "train": {"loss": 2.0},
            "validation": {"loss": 3.0},
            "validation_reward": 90.0,
        },
        {
            "epoch": 2,
            "train": {"loss": -0.5},
            "validation": {"loss": 1.0},
            "validation_reward": 95.0,
        },
    ]

    def check_plot(figure, path, **kwargs):
        assert path == tmp_path / "loss_curve.png"
        axes = figure.axes[0]
        lines = axes.get_lines()
        assert list(lines[0].get_xdata()) == [1, 2]
        assert list(lines[0].get_ydata()) == [2.0, -0.5]
        if stage == "xe":
            assert len(lines) == 2
            assert list(lines[1].get_ydata()) == [3.0, 1.0]
        else:
            assert len(lines) == 1
            assert axes.get_ylim()[0] < -0.5

    monkeypatch.setattr(Figure, "savefig", check_plot)
    export_loss_curve(history, Path(tmp_path), stage=stage)
