"""Typed experiment configuration loaded from TOML."""

from __future__ import annotations

import dataclasses
import tomllib
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str = "google/vit-base-patch16-384"
    encoder_revision: str = "2960116e809e2fca84146dbb240289aee7db4827"
    tokenizer_name: str = "distilbert-base-uncased"
    tokenizer_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    max_seq_length: int = 80
    d_model: int = 768
    num_heads: int = 12
    feedforward_dim: int = 1536
    num_decoder_layers: int = 4
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "relu"
    freeze_encoder: bool = True


@dataclass(frozen=True)
class DataConfig:
    images_dir: str = "data/flickr8k/Images"
    captions_file: str = "data/flickr8k/captions.txt"
    split_manifest: str = "data/splits/flickr8k.json"
    min_caption_words: int = 1
    expected_references_per_image: int = 5
    num_workers: int = 2
    horizontal_flip_probability: float = 0.4
    rotation_degrees: float = 10.0
    color_jitter_probability: float = 0.3
    brightness: float = 0.3
    hue: float = 0.1


@dataclass(frozen=True)
class XEConfig:
    epochs: int = 15
    batch_size: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    patience: int = 2


@dataclass(frozen=True)
class SCSTConfig:
    epochs: int = 8
    batch_size: int = 12
    learning_rate: float = 5e-5
    lr_decay: float = 0.85
    gradient_clip_norm: float = 1.0
    reward: Literal["nltk_meteor", "cider_d"] = "cider_d"
    baseline: Literal["greedy"] = "greedy"
    sample_model_mode: Literal["train", "eval"] = "eval"


@dataclass(frozen=True)
class EvaluationConfig:
    beam_size: int = 3
    length_penalty_alpha: float = 0.7
    include_spice: bool = False
    metrics_backend: Literal["coco"] = "coco"


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"
    amp: bool = True
    output_dir: str = "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    xe: XEConfig = field(default_factory=XEConfig)
    scst: SCSTConfig = field(default_factory=SCSTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.model.max_seq_length < 2:
            raise ValueError("model.max_seq_length must be at least 2")
        if not self.model.encoder_revision or not self.model.tokenizer_revision:
            raise ValueError("model encoder/tokenizer revisions must be pinned")
        if (
            self.model.d_model <= 0
            or self.model.d_model % 2
            or self.model.num_heads <= 0
            or self.model.d_model % self.model.num_heads
        ):
            raise ValueError("model.d_model must be positive and divisible by model.num_heads")
        if self.model.num_decoder_layers < 1 or self.model.feedforward_dim < 1:
            raise ValueError("model decoder layers/feedforward dimension must be positive")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.model.activation not in {"relu", "gelu"}:
            raise ValueError("model.activation must be 'relu' or 'gelu'")
        if self.data.expected_references_per_image < 1:
            raise ValueError("data.expected_references_per_image must be positive")
        if self.data.min_caption_words < 1:
            raise ValueError("data.min_caption_words must be positive")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        for name, value in (
            ("horizontal_flip_probability", self.data.horizontal_flip_probability),
            ("color_jitter_probability", self.data.color_jitter_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"data.{name} must be in [0, 1]")
        if self.data.rotation_degrees < 0 or self.data.brightness < 0:
            raise ValueError("data rotation_degrees/brightness cannot be negative")
        if not 0.0 <= self.data.hue <= 0.5:
            raise ValueError("data.hue must be in [0, 0.5]")
        if self.xe.epochs < 1 or self.xe.batch_size < 1:
            raise ValueError("xe.epochs and xe.batch_size must be positive")
        if self.xe.learning_rate <= 0 or self.xe.weight_decay < 0:
            raise ValueError("xe learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.xe.warmup_ratio < 1.0:
            raise ValueError("xe.warmup_ratio must be in [0, 1)")
        if self.xe.gradient_clip_norm <= 0 or self.xe.patience < 1:
            raise ValueError("xe gradient_clip_norm/patience must be positive")
        if self.scst.epochs < 1 or self.scst.batch_size < 1:
            raise ValueError("scst.epochs and scst.batch_size must be positive")
        if self.scst.learning_rate <= 0 or not 0.0 < self.scst.lr_decay <= 1.0:
            raise ValueError("scst learning_rate must be positive and lr_decay in (0, 1]")
        if self.scst.gradient_clip_norm <= 0:
            raise ValueError("scst.gradient_clip_norm must be positive")
        if self.scst.reward not in {"nltk_meteor", "cider_d"}:
            raise ValueError("scst.reward must be 'nltk_meteor' or 'cider_d'")
        if self.scst.baseline != "greedy":
            raise ValueError("scst.baseline currently supports only 'greedy'")
        if self.scst.sample_model_mode not in {"train", "eval"}:
            raise ValueError("scst.sample_model_mode must be 'train' or 'eval'")
        if self.evaluation.beam_size < 1:
            raise ValueError("evaluation.beam_size must be positive")
        if self.evaluation.length_penalty_alpha < 0:
            raise ValueError("evaluation.length_penalty_alpha cannot be negative")
        if self.evaluation.metrics_backend != "coco":
            raise ValueError("evaluation.metrics_backend currently supports only 'coco'")
        if self.runtime.seed < 0 or not self.runtime.device:
            raise ValueError("runtime.seed must be non-negative and device must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_T = TypeVar("_T")


def _from_mapping(cls: type[_T], values: dict[str, Any], section: str) -> _T:
    fields = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - fields)
    if unknown:
        raise ValueError(f"Unknown keys in [{section}]: {', '.join(unknown)}")
    annotations = get_type_hints(cls)
    for name, value in values.items():
        if not _matches_type(value, annotations[name]):
            raise TypeError(
                f"[{section}].{name} has type {type(value).__name__}; expected {annotations[name]}"
            )
    return cls(**values)


def _matches_type(value: Any, annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Literal:
        return any(
            type(value) is type(option) and value == option for option in get_args(annotation)
        )
    if origin in {Union, types.UnionType}:
        return any(_matches_type(value, option) for option in get_args(annotation))
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    return isinstance(value, annotation)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a strict TOML configuration file and validate its values."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    section_types = get_type_hints(ExperimentConfig)
    valid_sections = set(section_types)
    unknown_sections = sorted(set(raw) - valid_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {', '.join(unknown_sections)}")

    kwargs: dict[str, Any] = {}
    for section, section_type in section_types.items():
        if section in raw:
            values = raw[section]
            if not isinstance(values, dict):
                raise ValueError(f"[{section}] must be a TOML table")
            kwargs[section] = _from_mapping(section_type, values, section)

    config = ExperimentConfig(**kwargs)
    config.validate()
    return config
