from __future__ import annotations

import math

import pytest

from scst_captioner import rewards


def test_nltk_meteor_is_explicitly_named_and_uses_every_reference(monkeypatch) -> None:
    calls: list[tuple[tuple[tuple[str, ...], ...], tuple[str, ...], object]] = []
    injected_wordnet = object()

    def fake_meteor(image_references, prediction, **kwargs):
        calls.append((tuple(image_references), tuple(prediction), kwargs.get("wordnet")))
        return len(image_references) / 10 + len(prediction) / 100

    monkeypatch.setattr(rewards, "_load_nltk_meteor_score", lambda: fake_meteor)

    scores = rewards.nltk_meteor_rewards(
        [("a", "cat", "<eos>"), "a dog runs <eos>"],
        [
            [("a", "cat", "<eos>"), ("one", "cat", "<eos>")],
            ["a dog runs <eos>", "a running dog <eos>"],
        ],
        expected_references_per_image=2,
        wordnet=injected_wordnet,
    )

    assert rewards.NLTK_METEOR_REWARD_NAME == "nltk_meteor"
    assert scores == pytest.approx((0.23, 0.24))
    assert calls == [
        (
            (("a", "cat", "<eos>"), ("one", "cat", "<eos>")),
            ("a", "cat", "<eos>"),
            injected_wordnet,
        ),
        (
            (("a", "dog", "runs", "<eos>"), ("a", "running", "dog", "<eos>")),
            ("a", "dog", "runs", "<eos>"),
            injected_wordnet,
        ),
    ]


def test_nltk_meteor_casefolds_hypothesis_and_references_symmetrically(
    monkeypatch,
) -> None:
    observed = []

    def fake_meteor(image_references, prediction, **kwargs):
        del kwargs
        observed.append((image_references, prediction))
        return 1.0

    monkeypatch.setattr(rewards, "_load_nltk_meteor_score", lambda: fake_meteor)
    rewards.nltk_meteor_rewards(
        ["A CAT"],
        [["A Cat", "THE CAT"]],
        expected_references_per_image=2,
        wordnet=object(),
    )
    assert observed == [((("a", "cat"), ("the", "cat")), ("a", "cat"))]


def test_nltk_meteor_retains_empty_prediction_as_zero_reward(monkeypatch) -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("empty predictions do not need metric resources")

    monkeypatch.setattr(rewards, "_load_nltk_meteor_score", lambda: should_not_run)
    assert rewards.nltk_meteor_rewards(
        [""],
        [["a reference", "another reference"]],
        expected_references_per_image=2,
    ) == (0.0,)


@pytest.mark.parametrize(
    ("predictions", "references", "message"),
    [
        (["one"], [], "same number of images"),
        (["one"], [["only one"]], "expected 2"),
        (["one"], [["", "valid"]], "cannot be empty"),
    ],
)
def test_reward_inputs_reject_missing_or_incomplete_reference_sets(
    monkeypatch, predictions, references, message
) -> None:
    monkeypatch.setattr(rewards, "_load_nltk_meteor_score", lambda: None)
    with pytest.raises(rewards.RewardInputError, match=message):
        rewards.nltk_meteor_rewards(
            predictions,
            references,
            expected_references_per_image=2,
        )


def test_nltk_meteor_reports_missing_wordnet_without_downloading(monkeypatch) -> None:
    def missing_wordnet(*args, **kwargs):
        raise LookupError("wordnet not installed")

    monkeypatch.setattr(rewards, "_load_nltk_meteor_score", lambda: missing_wordnet)
    with pytest.raises(rewards.RewardDependencyError, match="Install it explicitly"):
        rewards.nltk_meteor_rewards(
            ["a caption"],
            [["first reference", "second reference"]],
            expected_references_per_image=2,
        )


def test_cider_d_uses_all_references_and_stable_document_frequency(monkeypatch) -> None:
    instances: list[FakeCiderScorer] = []

    class FakeCiderScorer:
        def __init__(self, *, n, sigma) -> None:
            self.n = n
            self.sigma = sigma
            self.items: list[tuple[str | None, list[str]]] = []
            self.document_frequency = None
            self.ref_len = None
            instances.append(self)

        def __iadd__(self, item):
            self.items.append(item)
            return self

        def compute_doc_freq(self) -> None:
            self.document_frequency = {("reference",): 2.0}

        def compute_cider(self):
            assert self.document_frequency == {("reference",): 2.0}
            assert self.ref_len == pytest.approx(math.log(2.0))
            return [0.4 + index / 10 for index in range(len(self.items))]

        def compute_score(self):
            raise AssertionError("global document frequencies should be used")

    monkeypatch.setattr(rewards, "_load_cider_d_scorer", lambda: FakeCiderScorer)

    scores = rewards.cider_d_rewards(
        ["sample one <eos>", "sample two <eos>"],
        [
            ["first reference <eos>", "second reference <eos>"],
            ["third reference <eos>", "fourth reference <eos>"],
        ],
        document_frequency_references=[
            ["corpus one reference", "corpus one alternate"],
            ["corpus two reference", "corpus two alternate"],
        ],
        expected_references_per_image=2,
    )

    assert rewards.CIDER_D_REWARD_NAME == "cider_d"
    assert scores == pytest.approx((0.4, 0.5))
    assert instances[0].items == [
        ("sample one <eos>", ["first reference <eos>", "second reference <eos>"]),
        ("sample two <eos>", ["third reference <eos>", "fourth reference <eos>"]),
    ]
    assert instances[1].items == [
        (None, ["corpus one reference", "corpus one alternate"]),
        (None, ["corpus two reference", "corpus two alternate"]),
    ]


def test_cached_cider_reward_normalizes_symmetrically_and_builds_df_once(
    monkeypatch,
) -> None:
    instances: list[object] = []

    class FakeCachedScorer:
        def __init__(self, *, n, sigma) -> None:
            del n, sigma
            self.items = []
            self.document_frequency = None
            self.ref_len = None
            self.doc_freq_calls = 0
            instances.append(self)

        def __iadd__(self, item):
            self.items.append(item)
            return self

        def compute_doc_freq(self) -> None:
            self.doc_freq_calls += 1
            self.document_frequency = {("cat",): 1.0}

        def compute_cider(self):
            assert self.document_frequency == {("cat",): 1.0}
            return [0.75] * len(self.items)

    monkeypatch.setattr(rewards, "_load_cider_d_scorer", lambda: FakeCachedScorer)
    reward = rewards.CiderDReward(
        [["A CAT. <eos>", "The Cat! <eos>"]],
        expected_references_per_image=2,
    )
    first = reward(["A Cat. <eos>"], [["A CAT. <eos>", "The Cat! <eos>"]])
    second = reward(["THE CAT! <eos>"], [["A CAT. <eos>", "The Cat! <eos>"]])

    assert first == second == (0.75,)
    assert instances[0].doc_freq_calls == 1
    assert instances[0].items == [(None, ["a cat <eos>", "the cat <eos>"])]
    assert instances[1].items == [("a cat <eos>", ["a cat <eos>", "the cat <eos>"])]
    assert instances[2].items == [("the cat <eos>", ["a cat <eos>", "the cat <eos>"])]
