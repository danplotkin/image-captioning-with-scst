from dataclasses import replace

import pytest
import torch
from test_model import MismatchedBackbone, TinyBackbone, tiny_config

from scst_captioner.generation import greedy_decode, sample_decode, scst_rollouts
from scst_captioner.losses import reinforce_loss
from scst_captioner.model import CPTRCaptioner


def make_model(encoder_hidden_size=4):
    torch.manual_seed(19)
    return CPTRCaptioner(
        replace(tiny_config(), num_decoder_layers=2),
        vocab_size=9,
        pad_token_id=0,
        backbone=TinyBackbone(hidden_size=encoder_hidden_size),
    ).eval()


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("encoder_hidden_size", [3, 4, 7])
def test_cached_prefix_logits_and_gradients(batch_size, encoder_hidden_size):
    model = make_model(encoder_hidden_size)
    tokens = torch.tensor([[1, 4, 0, 5, 2, 2], [1, 6, 7, 0, 0, 2], [1, 0, 4, 8, 5, 2]])[:batch_size]
    memory = torch.randn(batch_size, 4, encoder_hidden_size, requires_grad=True)
    results = []
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        memory.grad = None
        cache = model.init_decode_cache(memory) if cached else None
        logits = []
        for step in range(tokens.size(1)):
            if cached:
                output, cache = model.decode_step(tokens[:, step : step + 1], cache)
            else:
                output = model.predict_next_token(tokens[:, : step + 1], memory)
            logits.append(output)
        outputs = torch.stack(logits, dim=1)
        selected = outputs.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        mask = torch.tensor([[True, True, True, True, True, False]]).expand(batch_size, -1)
        loss = reinforce_loss(torch.linspace(-1, 1, batch_size), selected, mask)
        loss.backward()
        results.append(
            (
                outputs.detach(),
                loss.detach(),
                memory.grad.clone(),
                {
                    name: p.grad.clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                },
            )
        )
    for left, right in zip(results[0][:3], results[1][:3], strict=True):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert results[0][3].keys() == results[1][3].keys()
    for name in results[0][3]:
        torch.testing.assert_close(results[0][3][name], results[1][3][name], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("decode", [greedy_decode, sample_decode])
@pytest.mark.parametrize("encoder_hidden_size", [3, 4, 7])
def test_generation_cache_equivalence(decode, encoder_hidden_size):
    model = make_model(encoder_hidden_size)
    memory = torch.randn(3, 4, encoder_hidden_size)
    outputs = []
    for cached in (False, True):
        torch.manual_seed(41)
        outputs.append(
            decode(
                model,
                memory,
                bos_token_id=1,
                eos_token_id=2,
                max_new_tokens=6,
                forbidden_token_ids=(0, 1),
                use_kv_cache=cached,
            )
        )
    assert torch.equal(outputs[0].token_ids, outputs[1].token_ids)
    assert torch.equal(outputs[0].action_mask, outputs[1].action_mask)
    if outputs[0].action_log_probs is not None:
        torch.testing.assert_close(outputs[0].action_log_probs, outputs[1].action_log_probs)
        assert (outputs[1].action_log_probs[~outputs[1].action_mask] == 0).all()


@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_rollouts_cache_preserves_encoder_gradients_and_modes(mode, freeze_encoder):
    torch.manual_seed(19)
    model = CPTRCaptioner(
        tiny_config(freeze_encoder=freeze_encoder),
        vocab_size=9,
        pad_token_id=0,
        backbone=MismatchedBackbone(),
    ).train()
    outputs = scst_rollouts(
        model,
        torch.randn(2, 3, 2, 2),
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=4,
        sample_model_mode=mode,
        forbidden_token_ids=(0, 1),
    )
    assert model.training
    assert outputs.baseline.action_log_probs is None
    outputs.sampled.action_log_probs.sum().backward()
    attention = model.decoder.layers[0].multihead_attn
    for weight in (attention.q_proj_weight, attention.k_proj_weight, attention.v_proj_weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.count_nonzero() > 0
    encoder_grad = model.backbone.backbone.projection.weight.grad
    if freeze_encoder:
        assert encoder_grad is None
    else:
        assert encoder_grad is not None
        assert torch.isfinite(encoder_grad).all()
        assert encoder_grad.count_nonzero() > 0


def test_train_mode_falls_back_and_cache_has_no_state_dict_keys():
    model = make_model()
    keys = set(model.state_dict())
    cache = model.init_decode_cache(torch.randn(1, 4, 4))
    model.decode_step(torch.tensor([[1]]), cache)
    assert set(model.state_dict()) == keys
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model.init_decode_cache(torch.randn(1, 4, 4))
    outputs = []
    for cached in (False, True):
        torch.manual_seed(42)
        outputs.append(
            sample_decode(
                model,
                torch.ones(1, 4, 4),
                bos_token_id=1,
                eos_token_id=2,
                max_new_tokens=4,
                use_kv_cache=cached,
            )
        )
    assert torch.equal(outputs[0].token_ids, outputs[1].token_ids)
    torch.testing.assert_close(
        outputs[0].action_log_probs, outputs[1].action_log_probs, rtol=0, atol=0
    )


@pytest.mark.parametrize("encoder_hidden_size", [3, 4, 7])
def test_cached_autocast_logits_and_backward_are_finite(encoder_hidden_size):
    model = make_model(encoder_hidden_size)
    memory = torch.randn(2, 4, encoder_hidden_size)
    tokens = torch.tensor([[1, 4, 5], [1, 3, 2]])
    with torch.autocast("cpu", dtype=torch.float16):
        cache = model.init_decode_cache(memory)
        cached_logits = []
        full_logits = []
        for step in range(tokens.size(1)):
            output, cache = model.decode_step(tokens[:, step : step + 1], cache)
            cached_logits.append(output)
            full_logits.append(model.predict_next_token(tokens[:, : step + 1], memory))
        cached_logits = torch.stack(cached_logits, 1)
        full_logits = torch.stack(full_logits, 1)
    torch.testing.assert_close(cached_logits, full_logits, atol=3e-3, rtol=3e-3)
    cached_logits.float().log_softmax(-1).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cache_rejects_position_overflow():
    model = make_model()
    cache = model.init_decode_cache(torch.randn(1, 4, 4))
    for _ in range(model.max_length):
        _, cache = model.decode_step(torch.tensor([[1]]), cache)
    with pytest.raises(ValueError, match="positional capacity"):
        model.decode_step(torch.tensor([[1]]), cache)
