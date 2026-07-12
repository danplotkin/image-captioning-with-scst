"""Strict, COCO-style caption evaluation with comparison provenance.

The public API is record based so duplicate image IDs remain observable and
can be rejected before conversion to dictionaries.  pycocoevalcap and its
Java-backed metrics are imported only when evaluation is explicitly invoked.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypeAlias

ImageId: TypeAlias = str | int

COCO_METRIC_NAMES = (
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
)


class EvaluationInputError(ValueError):
    """Raised when an evaluation corpus is incomplete or ambiguously aligned."""


class EvaluationDependencyError(RuntimeError):
    """Raised when the explicitly requested COCO backend cannot run."""


@dataclass(frozen=True)
class PredictionRecord:
    """Exactly one generated caption associated with one image."""

    image_id: ImageId
    caption: str


@dataclass(frozen=True)
class ReferenceRecord:
    """The complete ordered reference set associated with one image."""

    image_id: ImageId
    captions: Sequence[str]


@dataclass(frozen=True)
class ValidatedCaptionCorpus:
    """Predictions and references in the declared canonical image order."""

    image_ids: tuple[ImageId, ...]
    predictions: tuple[str, ...]
    references: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ImageMetricRecord:
    """Per-image metric values, aligned to ``ValidatedCaptionCorpus.image_ids``."""

    image_id: ImageId
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"image_id": self.image_id, "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class EvaluationResult:
    """Corpus scores, aligned per-image scores, and reproducibility metadata."""

    metrics: Mapping[str, float]
    per_image: tuple[ImageMetricRecord, ...]
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": dict(self.metrics),
            "per_image": [record.to_dict() for record in self.per_image],
            "provenance": dict(self.provenance),
        }


def _validate_image_id(image_id: object, *, field: str) -> ImageId:
    if isinstance(image_id, bool) or not isinstance(image_id, (str, int)):
        raise TypeError(f"{field} must be a string or integer (but not bool)")
    if isinstance(image_id, str) and not image_id:
        raise EvaluationInputError(f"{field} cannot be an empty string")
    return image_id


def _duplicates(image_ids: Sequence[ImageId]) -> list[ImageId]:
    seen: set[ImageId] = set()
    reported: set[ImageId] = set()
    duplicates: list[ImageId] = []
    for image_id in image_ids:
        if image_id in seen and image_id not in reported:
            duplicates.append(image_id)
            reported.add(image_id)
        seen.add(image_id)
    return duplicates


def _format_ids(image_ids: Sequence[ImageId]) -> str:
    return ", ".join(repr(image_id) for image_id in image_ids)


def validate_caption_records(
    expected_image_ids: Sequence[ImageId],
    predictions: Sequence[PredictionRecord],
    references: Sequence[ReferenceRecord],
    *,
    expected_references_per_image: int | None = 5,
) -> ValidatedCaptionCorpus:
    """Validate one prediction and a complete reference set for every image.

    The output always follows ``expected_image_ids`` even when the two record
    sequences arrive in another order.  Unlike a mapping API, the record API
    can detect duplicate IDs rather than silently overwriting them.
    """

    if isinstance(expected_image_ids, str):
        raise TypeError("expected_image_ids must be a sequence of image IDs")
    canonical_ids = tuple(
        _validate_image_id(image_id, field=f"expected_image_ids[{index}]")
        for index, image_id in enumerate(expected_image_ids)
    )
    if not canonical_ids:
        raise EvaluationInputError("expected_image_ids cannot be empty")
    duplicate_expected = _duplicates(canonical_ids)
    if duplicate_expected:
        raise EvaluationInputError(
            f"expected_image_ids contains duplicates: {_format_ids(duplicate_expected)}"
        )
    if expected_references_per_image is not None and expected_references_per_image < 1:
        raise EvaluationInputError("expected_references_per_image must be positive or None")

    prediction_ids = tuple(
        _validate_image_id(record.image_id, field=f"predictions[{index}].image_id")
        for index, record in enumerate(predictions)
    )
    reference_ids = tuple(
        _validate_image_id(record.image_id, field=f"references[{index}].image_id")
        for index, record in enumerate(references)
    )
    duplicate_predictions = _duplicates(prediction_ids)
    if duplicate_predictions:
        raise EvaluationInputError(
            f"predictions contains duplicate image IDs: {_format_ids(duplicate_predictions)}"
        )
    duplicate_references = _duplicates(reference_ids)
    if duplicate_references:
        raise EvaluationInputError(
            f"references contains duplicate image IDs: {_format_ids(duplicate_references)}"
        )

    expected_set = set(canonical_ids)
    prediction_set = set(prediction_ids)
    reference_set = set(reference_ids)
    problems: list[str] = []
    missing_predictions = [image_id for image_id in canonical_ids if image_id not in prediction_set]
    extra_predictions = [image_id for image_id in prediction_ids if image_id not in expected_set]
    missing_references = [image_id for image_id in canonical_ids if image_id not in reference_set]
    extra_references = [image_id for image_id in reference_ids if image_id not in expected_set]
    if missing_predictions:
        problems.append(f"missing predictions: {_format_ids(missing_predictions)}")
    if extra_predictions:
        problems.append(f"extra predictions: {_format_ids(extra_predictions)}")
    if missing_references:
        problems.append(f"missing references: {_format_ids(missing_references)}")
    if extra_references:
        problems.append(f"extra references: {_format_ids(extra_references)}")
    if problems:
        raise EvaluationInputError("; ".join(problems))

    predictions_by_id: dict[ImageId, str] = {}
    for index, record in enumerate(predictions):
        if not isinstance(record.caption, str):
            raise TypeError(f"predictions[{index}].caption must be a string")
        # An immediate-EOS generation is a valid empty caption and must remain
        # visible to the evaluator rather than being dropped as a missing row.
        predictions_by_id[record.image_id] = record.caption

    references_by_id: dict[ImageId, tuple[str, ...]] = {}
    for index, record in enumerate(references):
        if isinstance(record.captions, str) or not isinstance(record.captions, Sequence):
            raise TypeError(f"references[{index}].captions must be a sequence of strings")
        image_references = tuple(record.captions)
        if not image_references:
            raise EvaluationInputError(f"references[{index}].captions cannot be empty")
        if any(not isinstance(caption, str) for caption in image_references):
            raise TypeError(f"references[{index}].captions must contain only strings")
        if any(not caption.strip() for caption in image_references):
            raise EvaluationInputError(
                f"references[{index}].captions cannot contain empty captions"
            )
        if (
            expected_references_per_image is not None
            and len(image_references) != expected_references_per_image
        ):
            raise EvaluationInputError(
                f"references[{index}] contains {len(image_references)} captions; "
                f"expected {expected_references_per_image}"
            )
        references_by_id[record.image_id] = image_references

    return ValidatedCaptionCorpus(
        image_ids=canonical_ids,
        predictions=tuple(predictions_by_id[image_id] for image_id in canonical_ids),
        references=tuple(references_by_id[image_id] for image_id in canonical_ids),
    )


@dataclass(frozen=True)
class _CocoComponents:
    tokenizer_type: type[Any]
    bleu_type: type[Any]
    meteor_type: type[Any]
    rouge_type: type[Any]
    cider_type: type[Any]
    spice_type: type[Any] | None
    version: str


def _load_coco_components(*, include_spice: bool) -> _CocoComponents:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        Spice = None
        if include_spice:
            from pycocoevalcap.spice.spice import Spice
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise EvaluationDependencyError(
            "COCO caption evaluation requires pycocoevalcap. Install the project "
            "with `pip install -e '.[eval]'`."
        ) from error

    # PTBTokenizer and METEOR are Java-backed even when SPICE is disabled.
    # Check once up front so a missing macOS/Unix Java runtime cannot degrade
    # into empty tokenizer output and misleading scorer assertion failures.
    import subprocess

    try:
        java = subprocess.run(
            ["java", "-version"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:  # pragma: no cover - environment
        raise EvaluationDependencyError(
            "COCO caption evaluation requires a working Java runtime."
        ) from error
    if java.returncode != 0:  # pragma: no cover - environment
        details = java.stderr.decode(errors="replace").strip()
        suffix = f" Runtime output: {details}" if details else ""
        raise EvaluationDependencyError(
            f"COCO caption evaluation requires a working Java runtime.{suffix}"
        )

    try:
        backend_version = version("pycocoevalcap")
    except PackageNotFoundError:  # pragma: no cover - editable/vendored installs
        backend_version = "unknown"
    return _CocoComponents(
        tokenizer_type=PTBTokenizer,
        bleu_type=Bleu,
        meteor_type=Meteor,
        rouge_type=Rouge,
        cider_type=Cider,
        spice_type=Spice,
        version=backend_version,
    )


def _as_finite_float(value: object, *, metric: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{metric} returned a non-numeric score") from error
    if not math.isfinite(converted):
        raise RuntimeError(f"{metric} returned a non-finite score")
    return converted


def _as_score_list(value: object, *, metric: str, expected_length: int) -> list[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise RuntimeError(f"{metric} returned malformed per-image scores")
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise RuntimeError(f"{metric} returned malformed per-image scores") from error
    if len(values) != expected_length:
        raise RuntimeError(
            f"{metric} returned {len(values)} per-image scores; expected {expected_length}"
        )
    return values


def _spice_f_score(value: object) -> object:
    if isinstance(value, Mapping):
        all_scores = value.get("All")
        if isinstance(all_scores, Mapping) and "f" in all_scores:
            return all_scores["f"]
    return value


def _typed_image_id(image_id: ImageId) -> dict[str, object]:
    return {"type": type(image_id).__name__, "value": image_id}


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_context(context: Mapping[str, object] | None) -> dict[str, object]:
    if context is None:
        return {}
    if any(not isinstance(key, str) for key in context):
        raise TypeError("provenance context keys must be strings")
    try:
        serialized = json.dumps(
            dict(context),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError("provenance context must be JSON serializable") from error
    loaded = json.loads(serialized)
    if not isinstance(loaded, dict):  # defensive; context starts as a mapping
        raise TypeError("provenance context must serialize to an object")
    return loaded


def _build_provenance(
    corpus: ValidatedCaptionCorpus,
    *,
    components: _CocoComponents,
    metric_names: Sequence[str],
    include_spice: bool,
    context: Mapping[str, object] | None,
) -> dict[str, object]:
    typed_ids = [_typed_image_id(image_id) for image_id in corpus.image_ids]
    reference_payload = [
        {"image_id": typed_id, "captions": list(image_references)}
        for typed_id, image_references in zip(typed_ids, corpus.references, strict=True)
    ]
    prediction_payload = [
        {"image_id": typed_id, "caption": prediction}
        for typed_id, prediction in zip(typed_ids, corpus.predictions, strict=True)
    ]
    protocol_payload = {
        "schema_version": 1,
        "backend": "pycocoevalcap",
        "backend_version": components.version,
        "tokenizer": "PTBTokenizer",
        "metrics": list(metric_names),
        "include_spice": include_spice,
        "image_ids": typed_ids,
        "references": reference_payload,
    }
    reference_counts = sorted({len(captions) for captions in corpus.references})
    return {
        "schema_version": 1,
        "backend": "pycocoevalcap",
        "backend_version": components.version,
        "tokenizer": "PTBTokenizer",
        "metrics": list(metric_names),
        "include_spice": include_spice,
        "num_images": len(corpus.image_ids),
        "reference_counts": reference_counts,
        "ordered_image_ids_sha256": _sha256_json(typed_ids),
        "references_sha256": _sha256_json(reference_payload),
        "predictions_sha256": _sha256_json(prediction_payload),
        "comparison_protocol_sha256": _sha256_json(protocol_payload),
        "context": _validated_context(context),
    }


def evaluate_coco_caption_metrics(
    corpus: ValidatedCaptionCorpus,
    *,
    include_spice: bool = False,
    context: Mapping[str, object] | None = None,
) -> EvaluationResult:
    """Compute the standard corpus caption suite with pycocoevalcap.

    Public image IDs are replaced by ordered integer backend IDs so even
    scorers that internally sort keys return per-image scores in the declared
    order.  The IDs are mapped back before returning.

    SPICE is opt-in.  pycocoevalcap may download Stanford CoreNLP models when
    SPICE is constructed; this function never performs that work on import.
    """

    if not corpus.image_ids:
        raise EvaluationInputError("the validated corpus cannot be empty")
    if not (len(corpus.image_ids) == len(corpus.predictions) == len(corpus.references)):
        raise EvaluationInputError("the validated corpus fields are not aligned")

    components = _load_coco_components(include_spice=include_spice)
    annotation_references = {
        index: [{"caption": caption} for caption in image_references]
        for index, image_references in enumerate(corpus.references)
    }
    annotation_predictions = {
        index: [{"caption": prediction}] for index, prediction in enumerate(corpus.predictions)
    }

    try:
        try:
            tokenizer = components.tokenizer_type(verbose=False)
        except TypeError:  # pragma: no cover - compatibility with old forks
            tokenizer = components.tokenizer_type()
        tokenized_references = tokenizer.tokenize(annotation_references)
        tokenized_predictions = tokenizer.tokenize(annotation_predictions)

        scorer_specs: list[tuple[Any, tuple[str, ...], bool]] = [
            (components.bleu_type(4), COCO_METRIC_NAMES[:4], False),
            (components.meteor_type(), ("METEOR",), False),
            (components.rouge_type(), ("ROUGE_L",), False),
            (components.cider_type(), ("CIDEr",), False),
        ]
        if include_spice:
            if components.spice_type is None:
                raise EvaluationDependencyError("SPICE was requested but is unavailable")
            scorer_specs.append((components.spice_type(), ("SPICE",), True))

        corpus_metrics: dict[str, float] = {}
        per_image_metrics: list[dict[str, float]] = [{} for _ in range(len(corpus.image_ids))]
        for scorer, names, is_spice in scorer_specs:
            aggregate, per_image = scorer.compute_score(
                tokenized_references,
                tokenized_predictions,
            )
            if len(names) > 1:
                aggregate_values = _as_score_list(
                    aggregate,
                    metric=names[0],
                    expected_length=len(names),
                )
                per_metric_values = _as_score_list(
                    per_image,
                    metric=names[0],
                    expected_length=len(names),
                )
                for name, aggregate_value, raw_image_values in zip(
                    names, aggregate_values, per_metric_values, strict=True
                ):
                    corpus_metrics[name] = _as_finite_float(aggregate_value, metric=name)
                    image_values = _as_score_list(
                        raw_image_values,
                        metric=name,
                        expected_length=len(corpus.image_ids),
                    )
                    for output, image_value in zip(per_image_metrics, image_values, strict=True):
                        output[name] = _as_finite_float(image_value, metric=name)
            else:
                name = names[0]
                corpus_metrics[name] = _as_finite_float(aggregate, metric=name)
                image_values = _as_score_list(
                    per_image,
                    metric=name,
                    expected_length=len(corpus.image_ids),
                )
                for output, image_value in zip(per_image_metrics, image_values, strict=True):
                    if is_spice:
                        image_value = _spice_f_score(image_value)
                    output[name] = _as_finite_float(image_value, metric=name)
    except FileNotFoundError as error:
        raise EvaluationDependencyError(
            "The COCO tokenizer and METEOR/SPICE scorers require a Java runtime."
        ) from error

    metric_names = tuple(corpus_metrics)
    provenance = _build_provenance(
        corpus,
        components=components,
        metric_names=metric_names,
        include_spice=include_spice,
        context=context,
    )
    return EvaluationResult(
        metrics=corpus_metrics,
        per_image=tuple(
            ImageMetricRecord(image_id=image_id, metrics=metrics)
            for image_id, metrics in zip(corpus.image_ids, per_image_metrics, strict=True)
        ),
        provenance=provenance,
    )


def evaluate_coco_caption_records(
    expected_image_ids: Sequence[ImageId],
    predictions: Sequence[PredictionRecord],
    references: Sequence[ReferenceRecord],
    *,
    expected_references_per_image: int | None = 5,
    include_spice: bool = False,
    context: Mapping[str, object] | None = None,
) -> EvaluationResult:
    """Validate sequence records, then compute the COCO caption metric suite."""

    corpus = validate_caption_records(
        expected_image_ids,
        predictions,
        references,
        expected_references_per_image=expected_references_per_image,
    )
    return evaluate_coco_caption_metrics(
        corpus,
        include_spice=include_spice,
        context=context,
    )


__all__ = [
    "COCO_METRIC_NAMES",
    "EvaluationDependencyError",
    "EvaluationInputError",
    "EvaluationResult",
    "ImageId",
    "ImageMetricRecord",
    "PredictionRecord",
    "ReferenceRecord",
    "ValidatedCaptionCorpus",
    "evaluate_coco_caption_metrics",
    "evaluate_coco_caption_records",
    "validate_caption_records",
]
