from dataclasses import replace

import pytest

from scst_captioner.config import ExperimentConfig, load_config


def test_flickr30k_methods_share_thirty_epoch_early_stopping_recipe():
    scst = load_config("configs/flickr30k-scst.toml")
    reinforce = load_config("configs/flickr30k-reinforce.toml")
    assert scst.scst.epochs == reinforce.scst.epochs == 30
    assert scst.scst.patience == reinforce.scst.patience == 4
    assert scst.scst.baseline == "greedy"
    assert reinforce.scst.baseline == "none"
    assert replace(scst, scst=replace(scst.scst, baseline="none")) == reinforce


@pytest.mark.parametrize("patience", [0, -1, 1.5, True])
def test_rl_patience_rejects_invalid_values(patience):
    config = ExperimentConfig()
    with pytest.raises(ValueError, match=r"scst\.patience"):
        replace(config, scst=replace(config.scst, patience=patience)).validate()


def test_existing_configs_do_not_enable_early_stopping_implicitly():
    assert load_config("configs/default.toml").scst.patience is None


@pytest.mark.parametrize("weight_decay", [-0.01, float("nan"), float("inf")])
def test_rl_weight_decay_must_be_nonnegative_and_finite(weight_decay):
    config = ExperimentConfig()
    with pytest.raises(ValueError, match=r"scst\.weight_decay"):
        replace(config, scst=replace(config.scst, weight_decay=weight_decay)).validate()


@pytest.mark.parametrize("encoder_lr", [None, 0.02])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("baseline", ["greedy", "none"])
def test_rl_adamw_applies_decoupled_decay_to_each_group(encoder_lr, weight_decay, baseline):
    import torch
    from torch import nn

    from scst_captioner.config import SCSTConfig
    from scst_captioner.training.optimization import build_scst_optimizer

    model = nn.Module()
    model.backbone = nn.Linear(1, 1, bias=False)
    model.head = nn.Linear(1, 1, bias=False)
    model.register_parameter("frozen", nn.Parameter(torch.ones(1), requires_grad=False))
    config = SCSTConfig(
        learning_rate=0.1,
        encoder_learning_rate=encoder_lr,
        weight_decay=weight_decay,
        baseline=baseline,
    )
    optimizer = build_scst_optimizer(model, config)
    assert type(optimizer) is torch.optim.AdamW
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
    optimizer.step()
    # Zero gradients isolate decoupled weight decay from Adam's adaptive update.
    for group in optimizer.param_groups:
        for p in group["params"]:
            torch.testing.assert_close(p, torch.full_like(p, 2 * (1 - group["lr"] * weight_decay)))
    torch.testing.assert_close(model.frozen, torch.tensor([2.0]))
