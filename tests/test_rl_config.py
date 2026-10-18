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
