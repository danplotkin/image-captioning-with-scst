"""Image-level, multi-reference rewards for self-critical training.

Metric dependencies are imported only when their reward is called.  In
particular, importing this module never downloads NLTK data or SPICE models.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

CaptionLike: TypeAlias = str | Sequence[str]

NLTK_METEOR_REWARD_NAME = "nltk_meteor"
CIDER_D_REWARD_NAME = "cider_d"
CIDER_D_TOKENIZATION = "nltk_treebank_casefold_coco_punctuation_v2"

_COCO_PUNCTUATION = {
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
}


class RewardInputError(ValueError):
    """Raised when predictions and image-level references are misaligned."""


class RewardDependencyError(RuntimeError):
    """Raised when an explicitly requested reward dependency is unavailable."""


def _caption_tokens(
    caption: CaptionLike,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(caption, str):
        tokens = tuple(caption.split())
    elif isinstance(caption, Sequence):
        tokens = tuple(caption)
        if any(not isinstance(token, str) for token in tokens):
            raise TypeError(f"{field} must contain only string tokens")
        if any(not token or token.isspace() for token in tokens):
            raise RewardInputError(f"{field} cannot contain empty tokens")
    else:
        raise TypeError(f"{field} must be a string or a sequence of string tokens")

    if not tokens and not allow_empty:
        raise RewardInputError(f"{field} cannot be empty")
    return tuple(token.casefold() for token in tokens)


def _prepare_reward_inputs(
    predictions: Sequence[CaptionLike],
    references: Sequence[Sequence[CaptionLike]],
    *,
    expected_references_per_image: int | None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[tuple[str, ...], ...], ...]]:
    if len(predictions) != len(references):
        raise RewardInputError("predictions and references must contain the same number of images")
    if not predictions:
        raise RewardInputError("at least one image is required to compute a reward")
    if expected_references_per_image is not None and expected_references_per_image < 1:
        raise RewardInputError("expected_references_per_image must be positive or None")

    prepared_predictions: list[tuple[str, ...]] = []
    prepared_references: list[tuple[tuple[str, ...], ...]] = []
    for image_index, (prediction, image_references) in enumerate(
        zip(predictions, references, strict=True)
    ):
        prepared_predictions.append(
            _caption_tokens(
                prediction,
                field=f"prediction[{image_index}]",
                allow_empty=True,
            )
        )
        if not isinstance(image_references, Sequence) or isinstance(image_references, str):
            raise TypeError(f"references[{image_index}] must be a sequence of captions")
        if not image_references:
            raise RewardInputError(f"references[{image_index}] cannot be empty")
        if (
            expected_references_per_image is not None
            and len(image_references) != expected_references_per_image
        ):
            raise RewardInputError(
                f"references[{image_index}] contains {len(image_references)} captions; "
                f"expected {expected_references_per_image}"
            )
        prepared_references.append(
            tuple(
                _caption_tokens(
                    reference,
                    field=f"references[{image_index}][{reference_index}]",
                    allow_empty=False,
                )
                for reference_index, reference in enumerate(image_references)
            )
        )

    return tuple(prepared_predictions), tuple(prepared_references)


def _load_nltk_meteor_score() -> Callable[..., float]:
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RewardDependencyError(
            "The 'nltk_meteor' reward requires NLTK. Install the project dependencies."
        ) from error
    return meteor_score


def nltk_meteor_rewards(
    predictions: Sequence[CaptionLike],
    references: Sequence[Sequence[CaptionLike]],
    *,
    expected_references_per_image: int | None = 5,
    wordnet: Any | None = None,
) -> tuple[float, ...]:
    """Return one explicitly named NLTK METEOR reward per image.

    Every prediction is compared with *all* references for its image.  NLTK's
    multi-reference METEOR implementation selects the best reference score.
    Inputs may be whitespace-delimited strings or pre-tokenized sequences.
    Both hypotheses and references are case-folded symmetrically before the
    metric sees them; punctuation remains explicit in the whitespace tokens.

    No NLTK resource is downloaded automatically.  The default NLTK scorer
    requires the WordNet corpus when unmatched words reach synonym matching.
    Callers may inject a WordNet-compatible object for hermetic execution.

    Special tokens are not stripped.  This deliberately lets an SCST caller
    retain an explicit EOS marker in training-reward captions.
    """

    prepared_predictions, prepared_references = _prepare_reward_inputs(
        predictions,
        references,
        expected_references_per_image=expected_references_per_image,
    )
    meteor_score = _load_nltk_meteor_score()

    rewards: list[float] = []
    for image_references, prediction in zip(prepared_references, prepared_predictions, strict=True):
        if not prediction:
            rewards.append(0.0)
            continue
        kwargs = {} if wordnet is None else {"wordnet": wordnet}
        try:
            score = float(meteor_score(image_references, prediction, **kwargs))
        except LookupError as error:
            raise RewardDependencyError(
                "NLTK METEOR could not find its WordNet data. Install it explicitly "
                "outside the training process (for example, `python -m nltk.downloader "
                "wordnet`) or inject a WordNet-compatible object."
            ) from error
        if not math.isfinite(score):
            raise RuntimeError("NLTK METEOR returned a non-finite reward")
        rewards.append(score)
    return tuple(rewards)


def _load_cider_d_scorer() -> type[Any]:
    try:
        # This scorer contains the clipping and Gaussian length penalty that
        # distinguish the COCO CIDEr-D implementation.
        from scst_captioner._cider import CiderDScorer
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RewardDependencyError(
            "The 'cider_d' reward requires the optional evaluation dependency. "
            "Install the project with `pip install -e '.[eval]'`."
        ) from error
    return CiderDScorer


def _validate_cider_parameters(n: int, sigma: float) -> None:
    if type(n) is not int or n < 1:
        raise RewardInputError("n must be a positive integer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise RewardInputError("sigma must be finite and positive")


def _normalize_cider_caption(caption: CaptionLike) -> str:
    """Apply one declared, symmetric approximation of COCO PTB tokenization."""

    from nltk.tokenize import TreebankWordTokenizer

    text = caption if isinstance(caption, str) else " ".join(caption)
    text = text.strip().casefold()
    completed = text.endswith("<eos>")
    if completed:
        text = text[: -len("<eos>")].rstrip()
    tokens = [
        token for token in TreebankWordTokenizer().tokenize(text) if token not in _COCO_PUNCTUATION
    ]
    if completed:
        tokens.append("<eos>")
    return " ".join(tokens)


class CiderDReward:
    """Cached, symmetrically tokenized CIDEr-D reward for repeated SCST batches."""

    name = CIDER_D_REWARD_NAME
    tokenization = CIDER_D_TOKENIZATION

    def __init__(
        self,
        reference_corpus: Sequence[Sequence[CaptionLike]],
        *,
        expected_references_per_image: int | None = 5,
        n: int = 4,
        sigma: float = 6.0,
    ) -> None:
        _validate_cider_parameters(n, sigma)
        if not reference_corpus:
            raise RewardInputError("CIDEr-D reference corpus cannot be empty")
        prepared_corpus = _prepare_reference_corpus(
            reference_corpus,
            expected_references_per_image=expected_references_per_image,
        )
        self.expected_references_per_image = expected_references_per_image
        self.n = n
        self.sigma = sigma
        scorer_type = _load_cider_d_scorer()
        document_scorer = scorer_type(n=n, sigma=sigma)
        for image_references in prepared_corpus:
            document_scorer += (
                None,
                [_normalize_cider_caption(reference) for reference in image_references],
            )
        document_scorer.compute_doc_freq()
        self.document_frequency = document_scorer.document_frequency
        self.ref_len = math.log(float(len(reference_corpus)))

    def __call__(
        self,
        predictions: Sequence[CaptionLike],
        references: Sequence[Sequence[CaptionLike]],
    ) -> tuple[float, ...]:
        prepared_predictions, prepared_references = _prepare_reward_inputs(
            predictions,
            references,
            expected_references_per_image=self.expected_references_per_image,
        )
        scorer_type = _load_cider_d_scorer()
        scorer = scorer_type(n=self.n, sigma=self.sigma)
        for prediction, image_references in zip(
            prepared_predictions, prepared_references, strict=True
        ):
            scorer += (
                _normalize_cider_caption(prediction),
                [_normalize_cider_caption(reference) for reference in image_references],
            )
        scorer.document_frequency = self.document_frequency
        scorer.corpus_ref_len = self.ref_len
        scores = tuple(float(score) for score in scorer.compute_cider())
        if len(scores) != len(predictions) or any(not math.isfinite(score) for score in scores):
            raise RuntimeError("CIDEr-D returned invalid per-image rewards")
        return scores


def _prepare_reference_corpus(
    reference_corpus: Sequence[Sequence[CaptionLike]],
    *,
    expected_references_per_image: int | None,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    placeholders: list[tuple[str, ...]] = [("<document-frequency-only>",)] * len(reference_corpus)
    _, prepared = _prepare_reward_inputs(
        placeholders,
        reference_corpus,
        expected_references_per_image=expected_references_per_image,
    )
    return prepared


def cider_d_rewards(
    predictions: Sequence[CaptionLike],
    references: Sequence[Sequence[CaptionLike]],
    *,
    document_frequency_references: Sequence[Sequence[CaptionLike]] | None = None,
    expected_references_per_image: int | None = 5,
    n: int = 4,
    sigma: float = 6.0,
) -> tuple[float, ...]:
    """Return a low-level, stateless CIDEr-D reward for aligned images.

    ``document_frequency_references`` should contain all training-image
    references.  Supplying it makes inverse-document-frequency statistics
    stable across mini-batches.  If omitted, pycocoevalcap derives statistics
    from the current batch, which is useful for tests but not recommended for
    research training.

    This helper assumes its strings are already normalized and recomputes
    document frequencies when a corpus is supplied. Production SCST should
    construct :class:`CiderDReward` once instead. The optional dependency is
    loaded only when this function is called.
    """

    _validate_cider_parameters(n, sigma)
    prepared_predictions, prepared_references = _prepare_reward_inputs(
        predictions,
        references,
        expected_references_per_image=expected_references_per_image,
    )
    scorer_type = _load_cider_d_scorer()
    scorer = scorer_type(n=n, sigma=sigma)
    for prediction, image_references in zip(prepared_predictions, prepared_references, strict=True):
        scorer += (
            " ".join(prediction),
            [" ".join(reference) for reference in image_references],
        )

    if document_frequency_references is None:
        _, raw_scores = scorer.compute_score()
    else:
        prepared_corpus = _prepare_reference_corpus(
            document_frequency_references,
            expected_references_per_image=expected_references_per_image,
        )
        document_scorer = scorer_type(n=n, sigma=sigma)
        for image_references in prepared_corpus:
            document_scorer += (
                None,
                [" ".join(reference) for reference in image_references],
            )
        document_scorer.compute_doc_freq()
        scorer.document_frequency = document_scorer.document_frequency
        scorer.corpus_ref_len = math.log(float(len(prepared_corpus)))
        raw_scores = scorer.compute_cider()

    scores = tuple(float(score) for score in raw_scores)
    if len(scores) != len(prepared_predictions):
        raise RuntimeError("CIDEr-D returned a reward count that does not match the batch")
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("CIDEr-D returned a non-finite reward")
    return scores


__all__ = [
    "CIDER_D_REWARD_NAME",
    "CIDER_D_TOKENIZATION",
    "NLTK_METEOR_REWARD_NAME",
    "CaptionLike",
    "CiderDReward",
    "RewardDependencyError",
    "RewardInputError",
    "cider_d_rewards",
    "nltk_meteor_rewards",
]
