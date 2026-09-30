"""Numerical tests against the real COCO scorer, including singleton batches."""

from __future__ import annotations

import math

import pytest

from scst_captioner import rewards

coco = pytest.importorskip("pycocoevalcap.cider.cider_scorer")

REFERENCES = [
    ["a red cat sits on the mat <eos>", "the red cat sits on a mat <eos>"],
    ["a blue dog runs in the park <eos>", "the dog runs through a park <eos>"],
    ["a green bird flies in the sky <eos>", "the bird flies through the sky <eos>"],
]
PREDICTIONS = ["a red cat <eos>", "a blue dog runs in the park <eos>", ""]


def _corpus_scores(predictions, references):
    scorer = coco.CiderScorer()
    for prediction, image_references in zip(predictions, references, strict=True):
        scorer += prediction, image_references
    return scorer.compute_score()[1]


@pytest.mark.parametrize("batch_indices", [(0,), (1,), (2,), (2, 0), (0, 1, 2), (0, 1, 2, 0)])
@pytest.mark.parametrize("cached", [False, True])
def test_training_corpus_scores_match_coco_independently_of_batch(batch_indices, cached):
    expected = _corpus_scores(PREDICTIONS, REFERENCES)
    predictions = [PREDICTIONS[i] for i in batch_indices]
    references = [REFERENCES[i] for i in batch_indices]
    if cached:
        reward = rewards.CiderDReward(REFERENCES, expected_references_per_image=2)
        actual = reward(predictions, references)
    else:
        actual = rewards.cider_d_rewards(
            predictions,
            references,
            document_frequency_references=REFERENCES,
            expected_references_per_image=2,
        )
    assert actual == pytest.approx([expected[i] for i in batch_indices], abs=1e-12)


def test_perfect_mismatch_repetition_and_completion_rewards():
    references = [[row[0], row[0]] for row in REFERENCES]
    reward = rewards.CiderDReward(references, expected_references_per_image=2)
    perfect = references[0][0]
    predictions = [
        perfect,
        "",
        "unrelated words nowhere",
        perfect.removesuffix(" <eos>"),
        "a red cat " * 10 + "<eos>",
    ]
    scores = reward(predictions, [references[0]] * len(predictions))
    assert scores[0] == pytest.approx(10.0)
    assert scores[1] == scores[2] == 0.0
    assert 0.0 < scores[3] < scores[0]
    assert 0.0 <= scores[4] < scores[0]
    assert all(math.isfinite(score) for score in scores)


def test_corpus_and_batch_normalization_agree_for_unicode_and_punctuation():
    raw = [
        ["Straße CAT. <eos>", "A cat! <eos>"],
        ["the Straße dog <eos>", "A dog! <eos>"],
        ["a bird flies <eos>", "The bird <eos>"],
    ]
    normalized = [
        ["strasse cat <eos>", "a cat <eos>"],
        ["the strasse dog <eos>", "a dog <eos>"],
        ["a bird flies <eos>", "the bird <eos>"],
    ]
    reward = rewards.CiderDReward(raw, expected_references_per_image=2)
    expected = _corpus_scores([row[0] for row in normalized], normalized)
    assert reward([row[0] for row in raw], raw) == pytest.approx(expected, abs=1e-12)
    assert reward.document_frequency[("strasse",)] == 2.0


@pytest.mark.parametrize("corpus", [[[]], [[""]], ["not a reference list"]])
def test_cached_corpus_rejects_invalid_references(corpus):
    with pytest.raises((rewards.RewardInputError, TypeError)):
        rewards.CiderDReward(corpus, expected_references_per_image=None)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"sigma": 0}, {"sigma": float("nan")}])
def test_cached_scorer_rejects_invalid_parameters(kwargs):
    with pytest.raises(rewards.RewardInputError):
        rewards.CiderDReward(REFERENCES, expected_references_per_image=2, **kwargs)


def test_stateless_corpus_mode_matches_coco():
    actual = rewards.cider_d_rewards(PREDICTIONS, REFERENCES, expected_references_per_image=2)
    assert actual == pytest.approx(_corpus_scores(PREDICTIONS, REFERENCES), abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_configured_ngram_order_handles_long_captions(n):
    references = [[row[0]] for row in REFERENCES]
    reward = rewards.CiderDReward(references, expected_references_per_image=1, n=n)
    assert reward([references[0][0]], [references[0]]) == pytest.approx([10.0])


def test_held_out_references_do_not_recompute_training_statistics():
    reward = rewards.CiderDReward(REFERENCES, expected_references_per_image=2)
    prediction = "a red bird runs through the park <eos>"
    held_out = [prediction, "a green cat flies in the sky <eos>"]
    before = dict(reward.document_frequency)
    single = reward([prediction], [held_out])[0]
    repeated = reward([prediction] * 4, [held_out] * 4)
    assert repeated == pytest.approx([single] * 4, abs=1e-12)
    assert reward.ref_len == pytest.approx(math.log(len(REFERENCES)))
    # Upstream defaultdict lookups may insert zero-valued unseen ngrams.
    assert {key: value for key, value in reward.document_frequency.items() if value} == before
