import dataclasses

import pytest
import torch
from transformers import ViTConfig, ViTModel

from scst_captioner.cli import _validate_checkpoint_config
from scst_captioner.config import ExperimentConfig, ModelConfig, XEConfig, load_config
from scst_captioner.model import CPTRCaptioner
from scst_captioner.training.optimization import build_xe_optimizer


def make_model(blocks=2, final_norm=True):
    config = ModelConfig(
        d_model=8,
        num_heads=2,
        feedforward_dim=16,
        num_decoder_layers=1,
        max_seq_length=6,
        freeze_encoder=False,
        encoder_trainable_blocks=blocks,
        encoder_train_final_layer_norm=final_norm,
    )
    backbone = ViTModel(
        ViTConfig(
            image_size=8,
            patch_size=4,
            hidden_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.2,
        ),
        add_pooling_layer=False,
    )
    return CPTRCaptioner(config, vocab_size=9, pad_token_id=0, backbone=backbone)


def test_partial_unfreezing_updates_only_selected_parameters():
    model = make_model().train()
    vit = model.backbone.backbone
    assert not vit.embeddings.training
    assert not vit.encoder.layer[0].training
    assert vit.encoder.layer[1].training
    assert vit.layernorm.training
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = build_xe_optimizer(model, XEConfig(learning_rate=1e-3, encoder_learning_rate=1e-4))
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    logits = model(torch.randn(2, 3, 8, 8), torch.tensor([[1, 3], [1, 4]]))
    torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([2, 5])).backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    for parameter in (
        vit.encoder.layer[2].output.dense.weight,
        vit.layernorm.weight,
        model.fc.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.count_nonzero() > 0
    assert not torch.equal(
        vit.encoder.layer[2].output.dense.weight,
        before["backbone.backbone.encoder.layer.2.output.dense.weight"],
    )
    model.eval()
    assert all(not module.training for module in model.modules())


@pytest.mark.parametrize(
    "blocks,final_norm,frozen", [(0, False, True), (0, True, False), (3, False, False)]
)
def test_boundary_policies(blocks, final_norm, frozen):
    model = make_model(blocks, final_norm)
    assert model.encoder_is_frozen is frozen
    assert model.backbone.backbone.layernorm.weight.requires_grad is final_norm


def test_invalid_block_count():
    with pytest.raises(ValueError, match="must be in"):
        make_model(4)


def test_old_checkpoint_defaults_and_strict_resume():
    config = ExperimentConfig()
    old = config.to_dict()
    del old["model"]["encoder_trainable_blocks"]
    del old["model"]["encoder_train_final_layer_norm"]
    del old["xe"]["init_checkpoint"]
    del old["xe"]["encoder_learning_rate"]
    checkpoint = {"stage": "xe", "config": old}
    _validate_checkpoint_config(checkpoint, config, full_config=True)
    _validate_checkpoint_config(checkpoint, config)
    assert "encoder_trainable_blocks" not in old["model"]
    fine = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            freeze_encoder=False,
            encoder_trainable_blocks=2,
        ),
    )
    _validate_checkpoint_config(checkpoint, fine, allow_freezing_change=True)
    with pytest.raises(ValueError):
        _validate_checkpoint_config(checkpoint, fine, full_config=True)
    wrong = dataclasses.replace(fine, model=dataclasses.replace(fine.model, num_decoder_layers=2))
    with pytest.raises(ValueError):
        _validate_checkpoint_config(checkpoint, wrong, allow_freezing_change=True)


def test_existing_and_finetune_configs():
    original = load_config("configs/flickr30k.toml")
    fine = load_config("configs/flickr30k-xe-finetune.toml")
    assert original.model.freeze_encoder
    assert original.model.encoder_trainable_blocks is None
    assert original.xe.encoder_learning_rate is None
    assert fine.model.encoder_trainable_blocks == 4
    assert fine.data == original.data
    assert fine.runtime.output_dir != original.runtime.output_dir
    with pytest.raises(ValueError):
        dataclasses.replace(
            original,
            model=dataclasses.replace(
                original.model,
                encoder_trainable_blocks=2,
            ),
        ).validate()


@pytest.mark.parametrize("field", ["learning_rate", "encoder_learning_rate"])
@pytest.mark.parametrize("rate", [0.0, -1e-6, float("nan"), float("inf")])
def test_invalid_rl_learning_rates(field, rate):
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="scst"):
        dataclasses.replace(
            config, scst=dataclasses.replace(config.scst, **{field: rate})
        ).validate()


def test_rl_optimizer_groups_updates_and_decay():
    from scst_captioner.training.optimization import build_scst_optimizer

    configs = [load_config(f"configs/flickr30k-{name}.toml") for name in ("reinforce", "scst")]
    assert dataclasses.replace(configs[0].scst, baseline="greedy") == configs[1].scst
    for config in configs:
        assert config.model.encoder_trainable_blocks == 4
        assert config.scst.learning_rate == 1e-5
        assert config.scst.encoder_learning_rate == 1e-6
        model = make_model()
        optimizer = build_scst_optimizer(model, config.scst)
        encoder_ids = {id(p) for p in model.backbone.parameters()}
        grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
        assert all(id(p) not in encoder_ids for p in optimizer.param_groups[0]["params"])
        assert all(id(p) in encoder_ids for p in optimizer.param_groups[1]["params"])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scst.lr_decay)
        optimizer.step()
        for name, p in model.named_parameters():
            rate = (1e-6 if id(p) in encoder_ids else 1e-5) if p.requires_grad else 0.0
            torch.testing.assert_close(p, before[name] - rate, rtol=0, atol=1e-7)
        scheduler.step()
        assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([8.5e-6, 8.5e-7])
        legacy = build_scst_optimizer(model, ExperimentConfig().scst)
        assert len(legacy.param_groups) == 1
        frozen = build_scst_optimizer(make_model(0, False), config.scst)
        assert len(frozen.param_groups) == 1
        assert frozen.param_groups[0]["lr"] == 1e-5


def test_old_rl_config_compatibility_and_rate_changes_rejected():
    config = ExperimentConfig()
    old = config.to_dict()
    del old["scst"]["encoder_learning_rate"]
    checkpoint = {"stage": "scst", "config": old}
    _validate_checkpoint_config(checkpoint, config, full_config=True)
    assert "encoder_learning_rate" not in old["scst"]
    changed = dataclasses.replace(
        config, scst=dataclasses.replace(config.scst, encoder_learning_rate=1e-6)
    )
    with pytest.raises(ValueError, match="configuration"):
        _validate_checkpoint_config(checkpoint, changed, full_config=True)
