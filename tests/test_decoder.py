from dataclasses import replace

import pytest
import torch
from test_model import tiny_config
from torch import nn

from scst_captioner.decoder import CaptionDecoder


@pytest.mark.parametrize("encoder_hidden_size", [3, 4, 7])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("norm_first", [False, True])
def test_decoder_matches_torch_reference(encoder_hidden_size, training, activation, norm_first):
    """Check against an independent forward path, including dropout and backward."""
    torch.manual_seed(29)
    config = replace(tiny_config(), num_decoder_layers=2, activation=activation, dropout=0.2)
    decoder = CaptionDecoder(config, encoder_hidden_size)
    for layer in decoder.layers:
        layer.norm_first = norm_first
    decoder.norm = nn.LayerNorm(config.d_model)
    decoder.train(training)

    reference_layer = nn.TransformerDecoderLayer(
        d_model=config.d_model,
        nhead=config.num_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation=config.activation,
        batch_first=True,
        norm_first=norm_first,
    )
    if encoder_hidden_size != config.d_model:
        reference_layer.multihead_attn = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            kdim=encoder_hidden_size,
            vdim=encoder_hidden_size,
            batch_first=True,
        )
    reference = nn.TransformerDecoder(
        reference_layer, config.num_decoder_layers, norm=nn.LayerNorm(config.d_model)
    ).train(training)
    reference.load_state_dict(decoder.state_dict(), strict=True)

    target = torch.randn(2, 5, config.d_model)
    memory = torch.randn(2, 3, encoder_hidden_size)
    probe = torch.randn_like(target)
    padding = torch.tensor([[False, False, True, False, False], [False, False, False, True, True]])
    causal = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    allowed = ~(causal[None, None] | padding[:, None, None, :])

    def run(module, inputs, image_features):
        if module is decoder:
            return module(inputs, image_features, allowed)[0]
        return module(
            inputs,
            image_features,
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
            tgt_is_causal=True,
        )

    results = []
    for module in (reference, decoder):
        inputs = target.clone().requires_grad_(True)
        image_features = memory.clone().requires_grad_(True)
        torch.manual_seed(73)
        output = run(module, inputs, image_features)
        (output * probe).sum().backward()
        results.append(
            (
                output.detach(),
                inputs.grad.clone(),
                image_features.grad.clone(),
                {name: p.grad.clone() for name, p in module.named_parameters()},
            )
        )
    expected, actual = results
    for left, right in zip(expected[:3], actual[:3], strict=True):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    assert expected[3].keys() == actual[3].keys()
    for name in expected[3]:
        torch.testing.assert_close(expected[3][name], actual[3][name], atol=3e-6, rtol=3e-5)

    if not training:
        # Also cover PyTorch's no-gradient attention fast path used at inference.
        with torch.inference_mode():
            torch.testing.assert_close(
                run(reference, target, memory),
                run(decoder, target, memory),
                atol=3e-6,
                rtol=3e-5,
            )
