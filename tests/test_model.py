from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scst_captioner.config import ModelConfig
from scst_captioner.model import CPTRCaptioner


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.projection = nn.Linear(3, hidden_size)
        self.dropout = nn.Dropout(0.9)

    def forward(self, pixels: torch.Tensor) -> SimpleNamespace:
        flattened = pixels.flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=self.dropout(self.projection(flattened)))


class MismatchedBackbone(TinyBackbone):
    def __init__(self) -> None:
        super().__init__(hidden_size=3)


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


@pytest.mark.parametrize("encoder_hidden_size", [3, 4, 7])
@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_cross_attention_consumes_raw_encoder_features(
    encoder_hidden_size: int, freeze_encoder: bool
) -> None:
    torch.manual_seed(23)
    model = CPTRCaptioner(
        tiny_config(freeze_encoder=freeze_encoder),
        vocab_size=9,
        pad_token_id=0,
        backbone=TinyBackbone(hidden_size=encoder_hidden_size),
    ).eval()
    pixels = torch.randn(2, 3, 2, 2)
    memory = model.encode_images(pixels)
    assert memory.shape == (2, 4, encoder_hidden_size)
    torch.testing.assert_close(memory, model.backbone(pixels))
    assert not hasattr(model, "memory_projection")
    assert model.encoder_is_frozen is freeze_encoder

    logits = model.decode(torch.tensor([[1, 3, 0], [1, 4, 5]]), memory)
    assert logits.shape == (2, 3, 9)
    nn.functional.cross_entropy(logits[:, -1], torch.tensor([2, 6])).backward()
    for parameter in model.decoder.layers[0].multihead_attn.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    encoder_grad = model.backbone.backbone.projection.weight.grad
    if freeze_encoder:
        assert encoder_grad is None
    else:
        assert encoder_grad is not None
        assert torch.isfinite(encoder_grad).all()
        assert encoder_grad.count_nonzero() > 0


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
