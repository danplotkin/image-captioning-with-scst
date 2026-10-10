from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from scst_captioner.training import SCSTTrainer, XETrainer


class TinyCaptioner(nn.Module):
    encoder_is_frozen = True

    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(1, 5, bias=False)
        nn.init.zeros_(self.output.weight)
        self.output_dtypes: list[torch.dtype] = []

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)

    def predict_next_token(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        del token_ids
        logits = self.output(memory[:, 0])
        self.output_dtypes.append(logits.dtype)
        return logits

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.predict_next_token(input_ids, self.encode_images(pixel_values))
        return logits[:, None, :].expand(-1, input_ids.size(1), -1)


class TinyTokenizer:
    bos_token_id = None
    eos_token_id = None
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    all_special_ids = (0, 1, 2, 4)

    @staticmethod
    def batch_decode(token_ids: torch.Tensor, **kwargs: Any) -> list[str]:
        del kwargs
        return ["word" if 3 in row.tolist() else "" for row in token_ids]


class UnitAdvantageReward:
    name = "unit_advantage"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, predictions: list[str], references: list[list[str]]) -> list[float]:
        del references
        self.calls += 1
        return [0.0 if self.calls % 2 else 1.0] * len(predictions)


def make_trainer(
    stage: str, *, amp: bool, growth_interval: int = 2000
) -> tuple[XETrainer | SCSTTrainer, TinyCaptioner, dict[str, Any]]:
    torch.manual_seed(7)
    model = TinyCaptioner()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    batch: dict[str, Any] = {"pixel_values": torch.ones(1, 3, 2, 2)}
    trainer: XETrainer | SCSTTrainer
    if stage == "xe":
        # The incorrect, confident prediction produces a finite loss but a
        # scaled FP16 derivative that exceeds 65504 at the initial scale.
        with torch.no_grad():
            model.output.weight[3] = -12.0
        trainer = XETrainer(
            model,
            optimizer,
            pad_token_id=0,
            device=torch.device("cpu"),
            scheduler=scheduler,
            amp=False,
        )
        batch.update(input_ids=torch.tensor([[1]]), labels=torch.tensor([[3]]))
    else:
        trainer = SCSTTrainer(
            model,
            optimizer,
            tokenizer=TinyTokenizer(),
            reward=UnitAdvantageReward(),
            device=torch.device("cpu"),
            max_new_tokens=1,
            scheduler=scheduler,
            amp=False,
        )
        batch.update(image_ids=["image.jpg"], references=[["word"]])
    if amp:
        # Production enables AMP on CUDA. CPU FP16 autocast and the real CPU
        # GradScaler exercise the same overflow path without requiring a GPU.
        trainer.amp_enabled = True
        trainer.scaler = torch.amp.GradScaler(
            "cpu", init_scale=65536.0, growth_interval=growth_interval
        )
    return trainer, model, batch


@pytest.mark.parametrize("stage", ["xe", "scst"])
def test_amp_overflow_skips_update_then_recovers(stage: str) -> None:
    trainer, model, batch = make_trainer(stage, amp=True, growth_interval=1)
    before = model.output.weight.detach().clone()
    initial_scheduler_epoch = trainer.scheduler.last_epoch

    skipped_metrics = trainer.train_epoch([batch])

    assert torch.isfinite(torch.tensor(skipped_metrics.loss))
    assert set(model.output_dtypes) == {torch.float16}
    assert not torch.isfinite(model.output.weight.grad).all()
    assert torch.equal(model.output.weight.detach(), before)
    assert not trainer.optimizer.state
    assert trainer.scaler.get_scale() == 32768.0
    assert trainer.global_step == 0
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    recovered_metrics = trainer.train_epoch([batch])

    assert torch.isfinite(torch.tensor(recovered_metrics.loss))
    assert torch.isfinite(model.output.weight.grad).all()
    assert not torch.equal(model.output.weight.detach(), before)
    assert trainer.optimizer.state[model.output.weight]["step"].item() == 1
    # A successful update that also grows the scale still counts as an update.
    assert trainer.scaler.get_scale() == 65536.0
    assert trainer.global_step == 1
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch + 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


@pytest.mark.parametrize("stage", ["xe", "scst"])
def test_amp_mixed_epoch_counts_only_successful_updates(stage: str) -> None:
    trainer, model, batch = make_trainer(stage, amp=True)
    initial_scheduler_epoch = trainer.scheduler.last_epoch

    trainer.train_epoch([batch, batch, batch])

    assert trainer.scaler.get_scale() == 32768.0
    assert trainer.global_step == 2
    assert trainer.optimizer.state[model.output.weight]["step"].item() == 2
    scheduler_steps = 2 if stage == "xe" else 1
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch + scheduler_steps


@pytest.mark.parametrize("stage", ["xe", "scst"])
@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")])
def test_full_precision_nonfinite_gradients_still_raise(stage: str, nonfinite: float) -> None:
    trainer, model, batch = make_trainer(stage, amp=False)
    before = model.output.weight.detach().clone()
    initial_scheduler_epoch = trainer.scheduler.last_epoch
    handle = model.output.weight.register_hook(
        lambda gradient: torch.full_like(gradient, nonfinite)
    )

    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.train_epoch([batch])
    finally:
        handle.remove()

    assert torch.equal(model.output.weight.detach(), before)
    assert not trainer.optimizer.state
    assert trainer.global_step == 0
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch


@pytest.mark.parametrize("stage", ["xe", "scst"])
def test_amp_finite_gradients_with_overflowing_norm_fail_before_update(stage: str) -> None:
    trainer, model, batch = make_trainer(stage, amp=True)
    before = model.output.weight.detach().clone()
    initial_scheduler_epoch = trainer.scheduler.last_epoch
    # These remain finite after unscaling, but squaring them in the float32
    # norm reduction overflows. GradScaler itself therefore does not skip.
    handle = model.output.weight.register_hook(lambda gradient: torch.full_like(gradient, 1e30))

    try:
        with pytest.raises(RuntimeError, match="gradient norm is non-finite despite finite"):
            trainer.train_epoch([batch])
    finally:
        handle.remove()

    assert torch.equal(model.output.weight.detach(), before)
    assert not trainer.optimizer.state
    assert trainer.scaler.get_scale() == 65536.0
    assert trainer.global_step == 0
    assert trainer.scheduler.last_epoch == initial_scheduler_epoch
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
