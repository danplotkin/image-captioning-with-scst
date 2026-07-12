from __future__ import annotations

import torch
from torch import nn

from scst_captioner.generation import (
    beam_decode,
    forbidden_special_ids,
    greedy_decode,
    length_penalty,
    sample_decode,
    scst_rollouts,
)


class ScriptedModel(nn.Module):
    encoder_is_frozen = True

    def __init__(self, vocab_size: int = 6) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, vocab_size))
        self.predict_modes: list[bool] = []
        self.encode_modes: list[bool] = []

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.encode_modes.append(self.training)
        return pixel_values[:, :1, :1, :1].reshape(pixel_values.size(0), 1, 1)

    def predict_next_token(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        self.predict_modes.append(self.training)
        step = min(token_ids.size(1) - 1, self.logits.size(0) - 1)
        output = self.logits[step].expand(token_ids.size(0), -1).clone()
        # Memory value 0 emits EOS immediately; value 1 emits token 3 then EOS.
        image_kind = memory[:, 0, 0].long()
        output[:, 2] -= 10.0
        output[:, 3] -= 10.0
        if step == 0:
            output[image_kind == 0, 2] += 30.0
            output[image_kind != 0, 3] += 30.0
        else:
            output[:, 2] += 30.0
        return output


def test_greedy_keeps_batch_rank_and_masks_after_first_eos() -> None:
    model = ScriptedModel()
    memory = torch.tensor([[[0.0]], [[1.0]]])
    output = greedy_decode(model, memory, bos_token_id=1, eos_token_id=2, max_new_tokens=4)
    assert output.token_ids.shape == (2, 3)
    assert output.action_mask.tolist() == [[True, False], [True, True]]
    assert output.token_ids[0].tolist() == [1, 2, 2]
    assert output.token_ids[1].tolist() == [1, 3, 2]


def test_sample_batch_one_is_two_dimensional_and_differentiable() -> None:
    torch.manual_seed(1)
    model = ScriptedModel()
    output = sample_decode(
        model, torch.tensor([[[0.0]]]), bos_token_id=1, eos_token_id=2, max_new_tokens=2
    )
    assert output.token_ids.ndim == 2
    assert output.action_log_probs is not None
    assert output.action_log_probs.ndim == 2
    assert output.action_mask.shape == output.action_log_probs.shape
    output.action_log_probs.masked_select(output.action_mask).sum().backward()
    assert model.logits.grad is not None


def test_scst_baseline_and_eval_sample_disable_dropout_and_restore_mode() -> None:
    torch.manual_seed(2)
    model = ScriptedModel()
    model.train()
    rollouts = scst_rollouts(
        model,
        torch.zeros(1, 1, 1, 1),
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=2,
        sample_model_mode="eval",
    )
    assert model.training is True
    assert model.encode_modes == [False]
    assert model.predict_modes and not any(model.predict_modes)
    assert rollouts.sampled.action_log_probs is not None
    rollouts.sampled.action_log_probs.masked_select(rollouts.sampled.action_mask).sum().backward()
    assert model.logits.grad is not None


def test_train_mode_sampling_is_explicit_but_baseline_stays_eval() -> None:
    torch.manual_seed(3)
    model = ScriptedModel()
    model.eval()
    scst_rollouts(
        model,
        torch.zeros(1, 1, 1, 1),
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=2,
        sample_model_mode="train",
    )
    assert model.training is False
    assert model.predict_modes[0] is False
    assert True in model.predict_modes[1:]


def test_special_tokens_are_forbidden_except_eos() -> None:
    ids = forbidden_special_ids([0, 1, 2, 5], bos_token_id=1, eos_token_id=2)
    assert ids == (0, 1, 5)
    model = ScriptedModel()
    with torch.no_grad():
        model.logits[0, 0] = 100.0
    output = greedy_decode(
        model,
        torch.tensor([[[1.0]]]),
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=1,
        forbidden_token_ids=ids,
    )
    assert output.action_token_ids.item() == 3


def test_beam_score_records_raw_log_probability_separately() -> None:
    model = ScriptedModel()
    output = beam_decode(
        model,
        torch.tensor([[[1.0]]]),
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=3,
        beam_size=2,
        length_penalty_alpha=0.7,
    )
    assert output.sequence_log_probs is not None
    assert output.sequence_log_probs.shape == (1,)
    assert length_penalty(1, 0.7) == 1.0
    assert length_penalty(3, 0.7) > 1.0
