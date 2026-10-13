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
    assert fine.model.encoder_trainable_blocks == 2
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
