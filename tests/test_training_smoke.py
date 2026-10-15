from __future__ import annotations

import pytest
import torch
from torch import nn

from scst_captioner.training import SCSTTrainer, XETrainer


class TinyXEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, 4)
        self.output = nn.Linear(4, 6)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        del pixel_values
        return self.output(self.embedding(input_ids))


class TinyPolicy(nn.Module):
    encoder_is_frozen = True

    def __init__(self) -> None:
        super().__init__()
        self.step_logits = nn.Parameter(
            torch.tensor([[-4.0, -4.0, 0.5, 1.0, -4.0], [-4.0, -4.0, 3.0, -1.0, -4.0]])
        )

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1, 1)

    def predict_next_token(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        del memory
        step = min(token_ids.size(1) - 1, 1)
        return self.step_logits[step].expand(token_ids.size(0), -1)


class TinyTokenizer:
    bos_token_id = None
    eos_token_id = None
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    all_special_ids = (0, 1, 2, 4)

    @staticmethod
    def batch_decode(
        token_ids: torch.Tensor, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ) -> list[str]:
        del skip_special_tokens, clean_up_tokenization_spaces
        return ["word" if 3 in row.tolist() else "" for row in token_ids]


class OrderedReward:
    name = "test_reward"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, predictions: object, references: object) -> list[float]:
        del references
        batch_size = len(predictions)  # type: ignore[arg-type]
        self.calls += 1
        return [0.0 if self.calls % 2 else 1.0] * batch_size


def test_vanilla_reinforce_uses_raw_reward_and_updates_policy() -> None:
    torch.manual_seed(7)
    policy = TinyPolicy()
    calls = []

    def reward(predictions, references):
        calls.append(predictions)
        return [2.0] * len(predictions)

    trainer = SCSTTrainer(
        policy,
        torch.optim.Adam(policy.parameters(), lr=0.01),
        tokenizer=TinyTokenizer(),
        reward=reward,
        device=torch.device("cpu"),
        max_new_tokens=2,
        baseline="none",
        amp=False,
    )
    before = policy.step_logits.detach().clone()
    metrics = trainer.train_epoch(
        [
            {
                "image_ids": ["image.jpg"],
                "pixel_values": torch.ones(1, 3, 2, 2),
                "references": [["one", "two"]],
            }
        ]
    )
    assert len(calls) == 1
    assert metrics.sampled_reward == metrics.advantage == 2.0
    assert metrics.baseline_reward == 0.0
    assert trainer.global_step == 1
    assert not torch.equal(before, policy.step_logits.detach())


@pytest.mark.parametrize("baseline", ["none", "greedy"])
def test_policy_gradient_trainer_matches_analytic_reward_weighted_update(
    monkeypatch: pytest.MonkeyPatch, baseline: str
) -> None:
    """Check the trainer's actual loss and update, including EOS/padding masking."""
    from scst_captioner.generation import GenerationOutput, SCSTRollouts

    policy = nn.Module()
    policy.log_probs = nn.Parameter(torch.tensor([[-0.2, -0.3, -9.0], [-0.4, -0.1, -8.0]]))
    mask = torch.tensor([[True, True, False], [True, True, False]])
    sampled = GenerationOutput(
        token_ids=torch.tensor([[1, 3, 2, 2], [1, 3, 2, 2]]),
        action_log_probs=policy.log_probs,
        action_mask=mask,
    )
    greedy = GenerationOutput(token_ids=sampled.token_ids, action_log_probs=None, action_mask=mask)

    def rollouts(*args, **kwargs):
        assert kwargs["baseline"] == baseline
        return SCSTRollouts(baseline=greedy if baseline == "greedy" else None, sampled=sampled)

    monkeypatch.setattr("scst_captioner.training.scst.scst_rollouts", rollouts)
    calls = []
    sampled_rewards = torch.tensor([2.0, -1.0], requires_grad=True)

    def reward(predictions, references):
        calls.append(predictions)
        if baseline == "greedy" and len(calls) == 1:
            return [0.5, 1.5]
        return sampled_rewards

    trainer = SCSTTrainer(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.1),
        tokenizer=TinyTokenizer(),
        reward=reward,
        device=torch.device("cpu"),
        max_new_tokens=3,
        baseline=baseline,
        amp=False,
        gradient_clip_norm=100.0,
    )
    before = policy.log_probs.detach().clone()
    metrics = trainer.train_epoch(
        [
            {
                "image_ids": ["a.jpg", "b.jpg"],
                "pixel_values": torch.ones(2, 3, 2, 2),
                "references": [["a"], ["b"]],
            }
        ]
    )
    weights = torch.tensor([2.0, -1.0] if baseline == "none" else [1.5, -2.5])
    expected_loss = -(weights * torch.tensor([-0.5, -0.5])).mean()
    expected_grad = -weights[:, None].expand_as(before) * mask / 2
    assert metrics.loss == pytest.approx(expected_loss.item())
    assert metrics.advantage == pytest.approx(weights.mean().item())
    assert len(calls) == (1 if baseline == "none" else 2)
    torch.testing.assert_close(policy.log_probs.grad, expected_grad)
    torch.testing.assert_close(policy.log_probs, before - 0.1 * expected_grad)
    assert sampled_rewards.grad is None


def test_tiny_cpu_xe_and_scst_optimizer_steps_complete() -> None:
    torch.manual_seed(7)
    xe_model = TinyXEModel()
    xe_optimizer = torch.optim.Adam(xe_model.parameters(), lr=0.01)
    xe_trainer = XETrainer(
        xe_model,
        xe_optimizer,
        pad_token_id=0,
        device=torch.device("cpu"),
        amp=False,
    )
    xe_batch = {
        "pixel_values": torch.zeros(2, 3, 2, 2),
        "input_ids": torch.tensor([[1, 3, 2], [1, 3, 0]]),
        "labels": torch.tensor([[3, 2, 0], [3, 2, 0]]),
    }
    train_metrics = xe_trainer.train_epoch([xe_batch])
    validation_metrics = xe_trainer.evaluate([xe_batch])
    assert train_metrics.tokens == 4
    assert validation_metrics.tokens == 4

    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    reward = OrderedReward()
    scst_trainer = SCSTTrainer(
        policy,
        optimizer,
        tokenizer=TinyTokenizer(),
        reward=reward,
        device=torch.device("cpu"),
        max_new_tokens=2,
        sample_model_mode="eval",
        amp=False,
    )
    before = policy.step_logits.detach().clone()
    metrics = scst_trainer.train_epoch(
        [
            {
                "image_ids": ["image.jpg"],
                "pixel_values": torch.ones(1, 3, 2, 2),
                "references": [["one", "two", "three", "four", "five"]],
            }
        ]
    )
    assert metrics.images == 1
    assert reward.calls == 2
    assert not torch.equal(before, policy.step_logits.detach())


def test_real_cider_scst_step_and_validation_batch_invariance() -> None:
    pytest.importorskip("pycocoevalcap")
    from dataclasses import replace
    from types import SimpleNamespace

    from scst_captioner.cli import _make_reward
    from scst_captioner.config import ExperimentConfig
    from scst_captioner.generation import GenerationOutput

    torch.manual_seed(0)
    config = ExperimentConfig()
    config = replace(config, data=replace(config.data, expected_references_per_image=2))
    references = [["word", "word"], ["other", "other"]]
    reward, include_eos = _make_reward(
        config, [SimpleNamespace(captions=row) for row in references]
    )
    assert include_eos is True
    policy = TinyPolicy()
    trainer = SCSTTrainer(
        policy,
        torch.optim.Adam(policy.parameters(), lr=0.01),
        tokenizer=TinyTokenizer(),
        reward=reward,
        device=torch.device("cpu"),
        max_new_tokens=2,
        include_eos_in_reward=include_eos,
        amp=False,
    )
    # Identical decoded text, but only the first sequence actually emitted EOS.
    output = GenerationOutput(
        token_ids=torch.tensor([[1, 3, 2], [1, 3, 3]]),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        action_log_probs=None,
    )
    assert trainer._decode_for_reward(output) == ["word <eos>", "word"]
    batch = {
        "image_ids": ["first.jpg", "second.jpg"],
        "pixel_values": torch.ones(2, 3, 2, 2),
        "references": references,
    }
    singleton_batches = [
        {
            "image_ids": [batch["image_ids"][i]],
            "pixel_values": batch["pixel_values"][i : i + 1],
            "references": [references[i]],
        }
        for i in range(2)
    ]
    assert trainer.evaluate([batch]) == pytest.approx(2.5)
    assert trainer.evaluate(singleton_batches) == pytest.approx(trainer.evaluate([batch]))
    before = policy.step_logits.detach().clone()
    metrics = trainer.train_epoch([batch])
    assert metrics.images == 2
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert metrics.advantage != 0.0
    assert not torch.equal(before, policy.step_logits.detach())
