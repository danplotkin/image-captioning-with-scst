from __future__ import annotations

import json
from typing import ClassVar

import pytest

from scst_captioner import evaluation


class FakeTokenizer:
    calls: ClassVar[list[dict[int, list[dict[str, str]]]]] = []

    def __init__(self, verbose=False) -> None:
        assert verbose is False

    def tokenize(self, annotations):
        self.calls.append(annotations)
        return {
            image_id: [item["caption"].lower() for item in captions]
            for image_id, captions in annotations.items()
        }


class FakeBleu:
    def __init__(self, n) -> None:
        assert n == 4

    def compute_score(self, references, predictions):
        _assert_backend_corpus(references, predictions)
        return (
            [0.1, 0.2, 0.3, 0.4],
            [
                [0.11, 0.12],
                [0.21, 0.22],
                [0.31, 0.32],
                [0.41, 0.42],
            ],
        )


class FakeMeteor:
    def compute_score(self, references, predictions):
        _assert_backend_corpus(references, predictions)
        return 0.5, [0.51, 0.52]


class FakeRouge:
    def compute_score(self, references, predictions):
        _assert_backend_corpus(references, predictions)
        return 0.6, [0.61, 0.62]


class FakeCider:
    def compute_score(self, references, predictions):
        _assert_backend_corpus(references, predictions)
        return 0.7, [0.71, 0.72]


class FakeSpice:
    def compute_score(self, references, predictions):
        _assert_backend_corpus(references, predictions)
        return 0.8, [
            {"All": {"f": 0.81, "p": 0.0, "r": 0.0}},
            {"All": {"f": 0.82, "p": 0.0, "r": 0.0}},
        ]


def _assert_backend_corpus(references, predictions) -> None:
    # Canonical integer IDs make backend sorting behavior deterministic.
    assert list(references) == [0, 1]
    assert list(predictions) == [0, 1]
    assert all(len(image_references) == 2 for image_references in references.values())
    assert all(len(image_predictions) == 1 for image_predictions in predictions.values())


def _components() -> evaluation._CocoComponents:
    return evaluation._CocoComponents(
        tokenizer_type=FakeTokenizer,
        bleu_type=FakeBleu,
        meteor_type=FakeMeteor,
        rouge_type=FakeRouge,
        cider_type=FakeCider,
        spice_type=FakeSpice,
        version="test-backend-1.0",
    )


def _records(prediction_b: str = "Prediction B"):
    expected_ids = ("image-b.jpg", "image-a.jpg")
    # Deliberately shuffled: expected_ids, not record arrival, defines alignment.
    predictions = (
        evaluation.PredictionRecord("image-a.jpg", "Prediction A"),
        evaluation.PredictionRecord("image-b.jpg", prediction_b),
    )
    references = (
        evaluation.ReferenceRecord("image-a.jpg", ("A ref one", "A ref two")),
        evaluation.ReferenceRecord("image-b.jpg", ("B ref one", "B ref two")),
    )
    return expected_ids, predictions, references


def test_validation_preserves_expected_order_and_all_references() -> None:
    expected_ids, predictions, references = _records()
    corpus = evaluation.validate_caption_records(
        expected_ids,
        predictions,
        references,
        expected_references_per_image=2,
    )

    assert corpus.image_ids == expected_ids
    assert corpus.predictions == ("Prediction B", "Prediction A")
    assert corpus.references == (
        ("B ref one", "B ref two"),
        ("A ref one", "A ref two"),
    )


@pytest.mark.parametrize(
    ("predictions", "references", "message"),
    [
        (
            (
                evaluation.PredictionRecord("image-b.jpg", "one"),
                evaluation.PredictionRecord("image-b.jpg", "duplicate"),
            ),
            (
                evaluation.ReferenceRecord("image-b.jpg", ("one", "two")),
                evaluation.ReferenceRecord("image-a.jpg", ("one", "two")),
            ),
            "duplicate image IDs",
        ),
        (
            (evaluation.PredictionRecord("image-b.jpg", "one"),),
            (
                evaluation.ReferenceRecord("image-b.jpg", ("one", "two")),
                evaluation.ReferenceRecord("image-a.jpg", ("one", "two")),
            ),
            "missing predictions",
        ),
        (
            (
                evaluation.PredictionRecord("image-b.jpg", "one"),
                evaluation.PredictionRecord("image-a.jpg", "two"),
                evaluation.PredictionRecord("unexpected.jpg", "three"),
            ),
            (
                evaluation.ReferenceRecord("image-b.jpg", ("one", "two")),
                evaluation.ReferenceRecord("image-a.jpg", ("one", "two")),
            ),
            "extra predictions",
        ),
        (
            (
                evaluation.PredictionRecord("image-b.jpg", "one"),
                evaluation.PredictionRecord("image-a.jpg", "two"),
            ),
            (
                evaluation.ReferenceRecord("image-b.jpg", ("only one",)),
                evaluation.ReferenceRecord("image-a.jpg", ("one", "two")),
            ),
            "expected 2",
        ),
    ],
)
def test_validation_rejects_duplicates_missing_extra_and_incomplete_references(
    predictions, references, message
) -> None:
    with pytest.raises(evaluation.EvaluationInputError, match=message):
        evaluation.validate_caption_records(
            ("image-b.jpg", "image-a.jpg"),
            predictions,
            references,
            expected_references_per_image=2,
        )


def test_validation_rejects_duplicate_expected_ids() -> None:
    with pytest.raises(evaluation.EvaluationInputError, match="expected_image_ids contains"):
        evaluation.validate_caption_records(
            ("same", "same"),
            (),
            (),
            expected_references_per_image=2,
        )


def test_coco_suite_returns_corpus_and_aligned_per_image_metrics(
    monkeypatch,
) -> None:
    FakeTokenizer.calls.clear()
    monkeypatch.setattr(
        evaluation,
        "_load_coco_components",
        lambda *, include_spice: _components(),
    )
    expected_ids, predictions, references = _records()

    result = evaluation.evaluate_coco_caption_records(
        expected_ids,
        predictions,
        references,
        expected_references_per_image=2,
        context={"checkpoint_stage": "scst", "decoder": "greedy", "seed": 42},
    )

    assert result.metrics == {
        "Bleu_1": 0.1,
        "Bleu_2": 0.2,
        "Bleu_3": 0.3,
        "Bleu_4": 0.4,
        "METEOR": 0.5,
        "ROUGE_L": 0.6,
        "CIDEr": 0.7,
    }
    assert [record.image_id for record in result.per_image] == list(expected_ids)
    assert result.per_image[0].metrics == {
        "Bleu_1": 0.11,
        "Bleu_2": 0.21,
        "Bleu_3": 0.31,
        "Bleu_4": 0.41,
        "METEOR": 0.51,
        "ROUGE_L": 0.61,
        "CIDEr": 0.71,
    }
    assert result.per_image[1].metrics["CIDEr"] == 0.72
    assert result.provenance["backend_version"] == "test-backend-1.0"
    assert result.provenance["reference_counts"] == [2]
    assert result.provenance["context"] == {
        "checkpoint_stage": "scst",
        "decoder": "greedy",
        "seed": 42,
    }
    # References and predictions both reach the tokenizer with all rows intact.
    assert len(FakeTokenizer.calls) == 2
    assert [len(rows) for rows in FakeTokenizer.calls[0].values()] == [2, 2]
    assert [len(rows) for rows in FakeTokenizer.calls[1].values()] == [1, 1]
    json.dumps(result.to_dict(), allow_nan=False)


def test_spice_is_optional_and_per_image_f_scores_remain_aligned(monkeypatch) -> None:
    requested: list[bool] = []

    def load(*, include_spice):
        requested.append(include_spice)
        return _components()

    monkeypatch.setattr(evaluation, "_load_coco_components", load)
    expected_ids, predictions, references = _records()
    result = evaluation.evaluate_coco_caption_records(
        expected_ids,
        predictions,
        references,
        expected_references_per_image=2,
        include_spice=True,
    )

    assert requested == [True]
    assert result.metrics["SPICE"] == 0.8
    assert [record.metrics["SPICE"] for record in result.per_image] == [0.81, 0.82]


def test_provenance_separates_protocol_identity_from_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "_load_coco_components",
        lambda *, include_spice: _components(),
    )
    expected_ids, predictions, references = _records("First B prediction")
    first = evaluation.evaluate_coco_caption_records(
        expected_ids,
        predictions,
        references,
        expected_references_per_image=2,
        context={"decoder": "greedy"},
    )
    expected_ids, predictions, references = _records("Second B prediction")
    second = evaluation.evaluate_coco_caption_records(
        expected_ids,
        predictions,
        references,
        expected_references_per_image=2,
        context={"decoder": "beam-3"},
    )

    assert (
        first.provenance["comparison_protocol_sha256"]
        == second.provenance["comparison_protocol_sha256"]
    )
    assert first.provenance["predictions_sha256"] != second.provenance["predictions_sha256"]
    assert first.provenance["references_sha256"] == second.provenance["references_sha256"]


def test_invalid_records_fail_before_optional_backend_is_loaded(monkeypatch) -> None:
    def must_not_load(*, include_spice):
        raise AssertionError("invalid input must fail before loading metric dependencies")

    monkeypatch.setattr(evaluation, "_load_coco_components", must_not_load)
    with pytest.raises(evaluation.EvaluationInputError, match="duplicate"):
        evaluation.evaluate_coco_caption_records(
            ("image",),
            (
                evaluation.PredictionRecord("image", "one"),
                evaluation.PredictionRecord("image", "two"),
            ),
            (evaluation.ReferenceRecord("image", ("one", "two")),),
            expected_references_per_image=2,
        )
