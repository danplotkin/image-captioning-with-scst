from __future__ import annotations

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
    def batch_decode(token_ids: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        del skip_special_tokens
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
