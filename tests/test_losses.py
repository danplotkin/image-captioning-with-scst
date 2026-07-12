import torch

from scst_captioner.losses import reinforce_loss, token_cross_entropy_sum


def test_reinforce_loss_uses_sequence_sum_and_advantage_sign() -> None:
    log_probs = torch.tensor(
        [[-0.2, -0.3, -9.0], [-0.4, -0.1, -8.0], [-0.7, -0.2, -7.0]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False], [True, True, False], [True, True, False]])
    advantage = torch.tensor([2.0, -1.0, 0.0], requires_grad=True)

    loss = reinforce_loss(advantage, log_probs, mask)

    assert torch.isclose(loss, torch.tensor(1.0 / 6.0))
    loss.backward()
    assert torch.allclose(log_probs.grad[0], torch.tensor([-2 / 3, -2 / 3, 0.0]))
    assert torch.allclose(log_probs.grad[1], torch.tensor([1 / 3, 1 / 3, 0.0]))
    assert torch.equal(log_probs.grad[2], torch.zeros(3))
    assert advantage.grad is None


def test_token_metrics_are_count_weighted() -> None:
    logits = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]], [[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]]
    )
    labels = torch.tensor([[0, 1, -1], [0, -1, -1]])
    loss, correct, count = token_cross_entropy_sum(logits, labels, pad_token_id=-1)
    assert loss.item() > 0
    assert correct == 2
    assert count == 3
