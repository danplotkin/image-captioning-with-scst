from types import SimpleNamespace

import torch
from torch import nn

from scst_captioner.config import ModelConfig
from scst_captioner.model import CPTRCaptioner


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.projection = nn.Linear(3, 4)
        self.dropout = nn.Dropout(0.9)

    def forward(self, pixels: torch.Tensor) -> SimpleNamespace:
        flattened = pixels.flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=self.dropout(self.projection(flattened)))


class MismatchedBackbone(TinyBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.projection = nn.Identity()


def tiny_config(*, freeze_encoder: bool = True) -> ModelConfig:
    return ModelConfig(
        encoder_name="unused",
        tokenizer_name="unused",
        max_seq_length=6,
        d_model=4,
        num_heads=2,
        feedforward_dim=8,
        num_decoder_layers=1,
        dropout=0.1,
        freeze_encoder=freeze_encoder,
    )


def test_instances_do_not_share_encoder_or_decoder_parameters() -> None:
    first = CPTRCaptioner(tiny_config(), vocab_size=9, pad_token_id=0, backbone=TinyBackbone())
    second = CPTRCaptioner(tiny_config(), vocab_size=9, pad_token_id=0, backbone=TinyBackbone())
    assert (
        first.decoder.layers[0].linear1.weight.data_ptr()
        != second.decoder.layers[0].linear1.weight.data_ptr()
    )
    assert (
        first.backbone.backbone.projection.weight.data_ptr()
        != second.backbone.backbone.projection.weight.data_ptr()
    )


def test_frozen_encoder_stays_in_eval_when_captioner_trains() -> None:
    model = CPTRCaptioner(
        tiny_config(freeze_encoder=True),
        vocab_size=9,
        pad_token_id=0,
        backbone=TinyBackbone(),
    )
    model.train()
    assert model.training
    assert not model.backbone.backbone.training


def test_trainable_memory_projection_keeps_scst_encoder_path_differentiable() -> None:
    model = CPTRCaptioner(
        tiny_config(freeze_encoder=True),
        vocab_size=9,
        pad_token_id=0,
        backbone=MismatchedBackbone(),
    )
    assert model.encoder_is_frozen is False


def test_teacher_forcing_shapes_and_legacy_key_paths() -> None:
    model = CPTRCaptioner(tiny_config(), vocab_size=9, pad_token_id=0, backbone=TinyBackbone())
    logits = model(torch.randn(2, 3, 2, 2), torch.tensor([[1, 3, 0], [1, 4, 5]]))
    assert logits.shape == (2, 3, 9)
    keys = set(model.state_dict())
    assert "backbone.backbone.projection.weight" in keys
    assert "positional_embedding.embedding.embedding.weight" in keys
    assert "positional_embedding.positional_encoding.pe" in keys
    assert "decoder.layers.0.linear1.weight" in keys
    assert "fc.weight" in keys
