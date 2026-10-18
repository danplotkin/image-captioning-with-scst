from unittest.mock import patch

import pytest
import torch
from test_generation import ScriptedModel
from test_kv_cache import make_model

from scst_captioner.generation import beam_decode, greedy_decode


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("alpha", [0.0, 0.7])
@pytest.mark.parametrize("hidden", [3, 4])
def test_beam_cache_matches_full_prefix(width, batch, alpha, hidden):
    model = make_model(hidden)
    memory = torch.randn(batch, 4, hidden)
    kwargs = dict(
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=6,
        beam_size=width,
        length_penalty_alpha=alpha,
        forbidden_token_ids=(0, 1),
    )
    with torch.inference_mode():
        expected = beam_decode(model, memory, **kwargs, use_kv_cache=False)
        with patch.object(model, "predict_next_token", side_effect=AssertionError("full prefix")):
            actual = beam_decode(model, memory, **kwargs)
    assert torch.equal(expected.token_ids, actual.token_ids)
    assert torch.equal(expected.action_mask, actual.action_mask)
    torch.testing.assert_close(expected.sequence_log_probs, actual.sequence_log_probs)
    assert not torch.isin(actual.action_token_ids[actual.action_mask], torch.tensor([0, 1])).any()
    if width == 1:
        with torch.inference_mode():
            greedy = greedy_decode(
                model,
                memory,
                bos_token_id=1,
                eos_token_id=2,
                max_new_tokens=6,
                forbidden_token_ids=(0, 1),
            )
        assert torch.equal(actual.token_ids, greedy.token_ids)


class CachedScriptedModel(ScriptedModel):
    def init_decode_cache(self, memory):
        return memory, torch.empty((1, 0), dtype=torch.long)

    def decode_step(self, tokens, cache):
        memory, prefix = cache
        assert tokens.shape == (1, 1)
        new_prefix = torch.cat([prefix, tokens], dim=1)
        return super().predict_next_token(new_prefix, memory), (memory, new_prefix)


def test_beam_cache_mixed_eos_and_no_cache_model_fallback():
    memory = torch.tensor([[[0.0]], [[1.0]]])
    kwargs = dict(bos_token_id=1, eos_token_id=2, max_new_tokens=4, beam_size=3)
    plain = ScriptedModel().eval()
    cached = CachedScriptedModel().eval()
    expected = beam_decode(plain, memory, **kwargs)
    with patch.object(cached, "predict_next_token", side_effect=AssertionError("full prefix")):
        actual = beam_decode(cached, memory, **kwargs)
    assert actual.token_ids.tolist() == [[1, 2, 2], [1, 3, 2]]
    assert actual.action_mask.tolist() == [[True, False], [True, True]]
    torch.testing.assert_close(actual.sequence_log_probs, expected.sequence_log_probs)


def test_beam_train_mode_does_not_initialize_cache():
    model = CachedScriptedModel().train()
    with patch.object(model, "init_decode_cache", side_effect=AssertionError("train cache")):
        beam_decode(
            model,
            torch.zeros(1, 1, 1),
            bos_token_id=1,
            eos_token_id=2,
            max_new_tokens=2,
            beam_size=3,
        )


def test_branch_caches_are_independent_and_earlier_winner_can_be_pruned():
    class Branches(torch.nn.Module):
        def predict_next_token(self, tokens, memory):
            prefix = tokens[0].tolist()
            # Token 3 leads initially, but token 4 has the better complete path.
            logits = {
                (1,): [-20.0, -20.0, -20.0, 2.0, 1.9],
                (1, 3): [0.0, 0.0, -10.0, 0.0, 0.0],
                (1, 4): [-20.0, -20.0, 10.0, -20.0, -20.0],
            }.get(tuple(prefix))
            return torch.tensor([logits or [-20.0, -20.0, 10.0, -20.0, -20.0]])

        def init_decode_cache(self, memory):
            return ()

        def decode_step(self, tokens, cache):
            prefix = (*cache, tokens.item())
            return self.predict_next_token(torch.tensor([prefix]), None), prefix

    model = Branches().eval()
    kwargs = dict(
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=3,
        beam_size=3,
        forbidden_token_ids=(0, 1),
        length_penalty_alpha=0.7,
    )
    expected = beam_decode(model, torch.zeros(1, 1, 1), **kwargs, use_kv_cache=False)
    actual = beam_decode(model, torch.zeros(1, 1, 1), **kwargs)
    assert actual.token_ids.tolist() == [[1, 4, 2]]
    assert torch.equal(expected.token_ids, actual.token_ids)
    torch.testing.assert_close(expected.sequence_log_probs, actual.sequence_log_probs)
