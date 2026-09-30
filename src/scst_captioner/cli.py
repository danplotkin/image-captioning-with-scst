"""Command-line workflows for data preparation, training, and evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from scst_captioner.artifacts import prepare_run_dir, read_json, write_json
from scst_captioner.checkpointing import (
    build_checkpoint,
    is_legacy_state_dict,
    load_checkpoint_file,
    load_model_checkpoint,
    restore_training_state,
    save_checkpoint,
    sha256_file,
)
from scst_captioner.config import ExperimentConfig, load_config
from scst_captioner.data import (
    SCSTImageDataset,
    XECaptionDataset,
    build_train_augmentation,
    collate_image_references,
    create_official_manifest,
    create_seeded_manifest,
    load_flickr8k_records,
    load_manifest,
    records_for_split,
    save_manifest,
    validate_manifest,
)
from scst_captioner.evaluation import (
    PredictionRecord,
    ReferenceRecord,
    evaluate_coco_caption_records,
)
from scst_captioner.generation import beam_decode, greedy_decode
from scst_captioner.model import build_model
from scst_captioner.reproducibility import (
    collect_environment,
    make_generator,
    resolve_device,
    restore_rng_state,
    seed_everything,
    seed_worker,
)
from scst_captioner.rewards import (
    CIDER_D_TOKENIZATION,
    CiderDReward,
    nltk_meteor_rewards,
)
from scst_captioner.tokenization import resolve_special_token_ids
from scst_captioner.training import SCSTTrainer, XETrainer


def _load_corpus(config: ExperimentConfig) -> tuple[Any, Any]:
    records = load_flickr8k_records(
        config.data.captions_file,
        config.data.images_dir,
        min_caption_words=config.data.min_caption_words,
        expected_references_per_image=config.data.expected_references_per_image,
        verify_images=True,
    )
    manifest = load_manifest(config.data.split_manifest)
    validate_manifest(
        manifest,
        records,
        provenance_files={"captions": config.data.captions_file},
    )
    return records, manifest


def _load_hf_components(config: ExperimentConfig) -> tuple[Any, Any]:
    from transformers import AutoTokenizer, ViTImageProcessor

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name, revision=config.model.tokenizer_revision
    )
    processor = ViTImageProcessor.from_pretrained(
        config.model.encoder_name, revision=config.model.encoder_revision
    )
    resolve_special_token_ids(tokenizer)
    return tokenizer, processor


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    config: ExperimentConfig,
    collate_fn: Any | None = None,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=make_generator(config.runtime.seed),
        pin_memory=resolve_device(config.runtime.device).type == "cuda",
        # Recreate workers each epoch so restoring the loader generator also
        # restores worker augmentation seeds exactly at epoch boundaries.
        persistent_workers=False,
    )


def _train_augmentation(config: ExperimentConfig) -> Any:
    return build_train_augmentation(
        horizontal_flip_probability=config.data.horizontal_flip_probability,
        rotation_degrees=config.data.rotation_degrees,
        color_jitter_probability=config.data.color_jitter_probability,
        brightness=config.data.brightness,
        hue=config.data.hue,
    )


def _run_provenance(
    config: ExperimentConfig,
    *,
    manifest: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "environment": collect_environment(Path.cwd()),
        "seed": config.runtime.seed,
        "deterministic": config.runtime.deterministic,
        "resolved_device": str(resolve_device(config.runtime.device)),
        "amp_requested": config.runtime.amp,
        "split_manifest": str(Path(config.data.split_manifest)),
        "split_manifest_sha256": sha256_file(config.data.split_manifest),
        "caption_records_sha256": manifest.records_sha256,
        "image_files_sha256": manifest.images_sha256,
        "encoder_name": config.model.encoder_name,
        "encoder_revision": config.model.encoder_revision,
        "tokenizer_name": config.model.tokenizer_name,
        "tokenizer_revision": config.model.tokenizer_revision,
        "freeze_encoder": config.model.freeze_encoder,
    }
    if extra:
        provenance.update(extra)
    return provenance


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_run_metadata(
    directory: Path,
    config: ExperimentConfig,
    provenance: dict[str, Any],
) -> None:
    write_json(directory / "resolved_config.json", config.to_dict())
    write_json(directory / "provenance.json", provenance)


def _validate_checkpoint_config(
    checkpoint: dict[str, Any],
    config: ExperimentConfig,
    *,
    expected_stage: str | None = None,
    full_config: bool = False,
    allow_legacy: bool = False,
) -> None:
    if checkpoint.get("legacy") or checkpoint.get("unverified_legacy"):
        if not allow_legacy:
            raise ValueError(
                "Raw legacy weights have unverified stage/configuration; use the explicit "
                "legacy override only for historical inspection"
            )
        return
    if expected_stage is not None and checkpoint.get("stage") != expected_stage:
        raise ValueError(
            f"Expected a {expected_stage!r} checkpoint, found {checkpoint.get('stage')!r}"
        )
    stored_config = checkpoint.get("config")
    expected_config = config.to_dict()
    expected = expected_config if full_config else expected_config["model"]
    actual = (
        stored_config
        if full_config
        else stored_config.get("model")
        if isinstance(stored_config, dict)
        else None
    )
    if actual != expected:
        scope = "experiment" if full_config else "model"
        raise ValueError(f"Checkpoint {scope} configuration does not match the requested config")


def _validate_checkpoint_data(
    checkpoint: dict[str, Any],
    config: ExperimentConfig,
    manifest: Any,
) -> None:
    if checkpoint.get("legacy") or checkpoint.get("unverified_legacy"):
        return
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint is missing data provenance")
    expected = {
        "split_manifest_sha256": sha256_file(config.data.split_manifest),
        "caption_records_sha256": manifest.records_sha256,
        "image_files_sha256": manifest.images_sha256,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match the current dataset protocol")


def _validate_checkpoint_scientific_protocol(
    checkpoint: dict[str, Any], config: ExperimentConfig
) -> None:
    if checkpoint.get("legacy") or checkpoint.get("unverified_legacy"):
        return
    stored = checkpoint.get("config")
    if not isinstance(stored, dict):
        raise ValueError("Checkpoint is missing its resolved configuration")
    stage = checkpoint.get("stage")
    if stage == "scst" and stored.get("scst") != config.to_dict()["scst"]:
        raise ValueError("SCST checkpoint reward/baseline/sampling protocol does not match config")
    if stage == "scst":
        provenance = checkpoint.get("provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "reward_protocol"
        ) != _reward_protocol_metadata(config):
            raise ValueError("SCST checkpoint reward implementation does not match environment")
    if stage == "xe" and stored.get("xe") != config.to_dict()["xe"]:
        raise ValueError("XE checkpoint training protocol does not match config")
    stored_data = stored.get("data")
    current_data = config.to_dict()["data"]
    preprocessing_keys = {
        "min_caption_words",
        "expected_references_per_image",
        "horizontal_flip_probability",
        "rotation_degrees",
        "color_jitter_probability",
        "brightness",
        "hue",
    }
    if not isinstance(stored_data, dict) or any(
        stored_data.get(key) != current_data.get(key) for key in preprocessing_keys
    ):
        raise ValueError("Checkpoint data/preprocessing protocol does not match config")


def _recorded_xe_parent_sha256(provenance: Any) -> str | None:
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("xe_checkpoint_sha256")
    if isinstance(value, str) and value:
        return value
    return _recorded_xe_parent_sha256(provenance.get("parent_provenance"))


def _preflight_checkpoint(
    path: str,
    *,
    expected_stage: str,
    config: ExperimentConfig,
    manifest: Any,
) -> dict[str, Any]:
    loaded = load_checkpoint_file(path, map_location="cpu")
    if is_legacy_state_dict(loaded):
        raise ValueError(
            f"The {expected_stage} matrix checkpoint is a raw, unverified legacy state dict"
        )
    _validate_checkpoint_config(loaded, config, expected_stage=expected_stage)
    _validate_checkpoint_data(loaded, config, manifest)
    _validate_checkpoint_scientific_protocol(loaded, config)
    summary = {
        "stage": loaded.get("stage"),
        "config": loaded.get("config", {}),
        "provenance": loaded.get("provenance", {}),
        "unverified_legacy": bool(loaded.get("unverified_legacy")),
    }
    del loaded
    return summary


def _resume_if_requested(
    path: str | None,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: ExperimentConfig,
    manifest: Any,
    expected_stage: str,
) -> tuple[int, int, dict[str, Any] | None]:
    if path is None:
        return 1, 0, None
    checkpoint = load_model_checkpoint(model, path, map_location="cpu")
    if checkpoint.get("legacy") or checkpoint.get("unverified_legacy"):
        raise ValueError("A raw legacy state dict can initialize training but cannot resume it")
    _validate_checkpoint_config(
        checkpoint,
        config,
        expected_stage=expected_stage,
        full_config=True,
    )
    _validate_checkpoint_data(checkpoint, config, manifest)
    required_resume_keys = {
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "training_state",
    }
    missing_resume_keys = sorted(required_resume_keys - set(checkpoint))
    if missing_resume_keys or not isinstance(checkpoint.get("training_state"), dict):
        if not missing_resume_keys:
            missing_resume_keys = ["valid training_state"]
        raise ValueError(
            "Checkpoint cannot resume training; missing " + ", ".join(missing_resume_keys)
        )
    restore_training_state(
        checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        strict=True,
    )
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    progress = checkpoint.get("progress", {})
    summary = {
        "best_metric": checkpoint.get("best_metric"),
        "training_state": checkpoint.get("training_state", {}),
        "provenance": checkpoint.get("provenance", {}),
        "schema_version": checkpoint.get("schema_version"),
        "stage": checkpoint.get("stage"),
    }
    return (
        int(progress.get("epoch", 0)) + 1,
        int(progress.get("global_step", 0)),
        summary,
    )


def _restore_loader_generator(loader: DataLoader[Any], state: Any) -> None:
    if state is None:
        return
    if loader.generator is None or not isinstance(state, torch.Tensor):
        raise ValueError("Checkpoint contains invalid DataLoader generator state")
    loader.generator.set_state(state.cpu())


def _best_metric_value(checkpoint: dict[str, Any] | None, default: float) -> float:
    best_metric = (checkpoint or {}).get("best_metric")
    if not isinstance(best_metric, dict) or "value" not in best_metric:
        return default
    return float(best_metric["value"])


def _resume_history(
    path: Path,
    checkpoint: dict[str, Any] | None,
    *,
    expected_last_epoch: int,
) -> list[dict[str, Any]]:
    checkpoint_history = (checkpoint or {}).get("training_state", {}).get("history", [])
    history = read_json(path) if path.exists() else checkpoint_history
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("Training history must be a list of objects")
    if history and int(history[-1].get("epoch", -1)) != expected_last_epoch:
        raise ValueError("Training history does not align with the resume checkpoint epoch")
    return list(history)


def command_prepare_data(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    records = load_flickr8k_records(
        config.data.captions_file,
        config.data.images_dir,
        min_caption_words=config.data.min_caption_words,
        expected_references_per_image=config.data.expected_references_per_image,
        verify_images=True,
    )
    official_paths = (args.official_train, args.official_validation, args.official_test)
    if any(official_paths) and not all(official_paths):
        raise ValueError("All three official split files must be supplied together")
    if all(official_paths):
        manifest = create_official_manifest(
            records,
            captions_file=config.data.captions_file,
            train_file=args.official_train,
            validation_file=args.official_validation,
            test_file=args.official_test,
            expected_references_per_image=config.data.expected_references_per_image,
            caption_policy=(
                f"minimum-{config.data.min_caption_words}-words-validation-no-filter-v1"
            ),
        )
    else:
        manifest = create_seeded_manifest(
            records,
            captions_file=config.data.captions_file,
            seed=config.runtime.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            expected_references_per_image=config.data.expected_references_per_image,
            caption_policy=(
                f"minimum-{config.data.min_caption_words}-words-validation-no-filter-v1"
            ),
        )
    output = Path(config.data.split_manifest)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Split manifest already exists: {output}")
    save_manifest(manifest, output)
    print(
        f"Wrote {manifest.strategy} image-level split to {output}: "
        f"train={len(manifest.train)}, validation={len(manifest.validation)}, "
        f"test={len(manifest.test)}, excluded={len(manifest.excluded)}"
    )
    return 0


def command_validate_data(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    records, manifest = _load_corpus(config)
    print(
        f"Validated {len(records)} images and {sum(len(item.captions) for item in records)} "
        f"captions against {config.data.split_manifest}."
    )
    print(
        f"train={len(manifest.train)}, validation={len(manifest.validation)}, "
        f"test={len(manifest.test)}, excluded={len(manifest.excluded)}"
    )
    return 0


def command_train_xe(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    records, manifest = _load_corpus(config)
    tokenizer, processor = _load_hf_components(config)
    train_records = records_for_split(records, manifest, "train", validate=False)
    validation_records = records_for_split(records, manifest, "validation", validate=False)
    train_dataset = XECaptionDataset(
        train_records,
        tokenizer=tokenizer,
        image_processor=processor,
        max_seq_length=config.model.max_seq_length,
        pil_transform=_train_augmentation(config),
    )
    validation_dataset = XECaptionDataset(
        validation_records,
        tokenizer=tokenizer,
        image_processor=processor,
        max_seq_length=config.model.max_seq_length,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=config.xe.batch_size,
        shuffle=True,
        config=config,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=config.xe.batch_size,
        shuffle=False,
        config=config,
    )
    model = build_model(config.model, tokenizer).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.xe.learning_rate,
        weight_decay=config.xe.weight_decay,
    )
    from transformers import get_cosine_schedule_with_warmup

    total_steps = len(train_loader) * config.xe.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.xe.warmup_ratio),
        num_training_steps=total_steps,
    )
    special_tokens = resolve_special_token_ids(tokenizer)
    trainer = XETrainer(
        model,
        optimizer,
        pad_token_id=special_tokens.pad,
        device=device,
        scheduler=scheduler,
        amp=config.runtime.amp,
        gradient_clip_norm=config.xe.gradient_clip_norm,
    )
    start_epoch, global_step, resume_checkpoint = _resume_if_requested(
        args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=trainer.scaler,
        config=config,
        manifest=manifest,
        expected_stage="xe",
    )
    _restore_loader_generator(
        train_loader,
        (resume_checkpoint or {}).get("training_state", {}).get("data_loader_generator_state"),
    )
    trainer.global_step = global_step
    run_dir = prepare_run_dir(
        args.output_dir or Path(config.runtime.output_dir) / "xe",
        overwrite=args.overwrite or args.resume is not None,
    )
    provenance = _run_provenance(
        config,
        manifest=manifest,
        extra={
            "stage": "xe",
            "resume_checkpoint": str(Path(args.resume)) if args.resume else None,
            "resume_checkpoint_sha256": sha256_file(args.resume) if args.resume else None,
            "parent_provenance": (resume_checkpoint or {}).get("provenance", {}),
        },
    )
    _write_run_metadata(run_dir, config, provenance)

    history_path = run_dir / "history.json"
    history: list[dict[str, Any]] = (
        _resume_history(
            history_path,
            resume_checkpoint,
            expected_last_epoch=start_epoch - 1,
        )
        if args.resume
        else []
    )
    best_loss = _best_metric_value(resume_checkpoint, float("inf"))
    stale_epochs = int((resume_checkpoint or {}).get("training_state", {}).get("stale_epochs", 0))
    for epoch in range(start_epoch, config.xe.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        validation_metrics = trainer.evaluate(validation_loader)
        row = {
            "epoch": epoch,
            "train": dataclasses.asdict(train_metrics),
            "validation": dataclasses.asdict(validation_metrics),
        }
        history.append(row)
        improved = validation_metrics.loss < best_loss
        if improved:
            best_loss = validation_metrics.loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = build_checkpoint(
            model,
            stage="xe",
            config=config.to_dict(),
            epoch=epoch,
            global_step=trainer.global_step,
            best_metric={"name": "validation_xe_loss", "value": best_loss, "mode": "min"},
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            provenance=provenance,
            training_state={
                "stale_epochs": stale_epochs,
                "history": history,
                "data_loader_generator_state": (
                    train_loader.generator.get_state()
                    if train_loader.generator is not None
                    else None
                ),
            },
        )
        save_checkpoint(checkpoint, run_dir / "last.pt")
        if improved:
            save_checkpoint(checkpoint, run_dir / "best.pt")
        write_json(run_dir / "history.json", history)
        print(
            f"XE epoch {epoch}: train_loss={train_metrics.loss:.4f}, "
            f"val_loss={validation_metrics.loss:.4f}, val_accuracy={validation_metrics.accuracy:.4f}"
        )
        if stale_epochs >= config.xe.patience:
            print(f"Early stopping after {stale_epochs} non-improving epoch(s).")
            break
    return 0


def _make_reward(config: ExperimentConfig, train_records: Sequence[Any]) -> tuple[Any, bool]:
    expected = config.data.expected_references_per_image
    if config.scst.reward == "nltk_meteor":
        from nltk.corpus import wordnet

        try:
            wordnet.ensure_loaded()
        except LookupError as error:
            raise RuntimeError(
                "The nltk_meteor reward requires WordNet. Run `uv run python -m "
                "nltk.downloader wordnet omw-1.4` before training or evaluation."
            ) from error
        return (
            functools.partial(
                nltk_meteor_rewards,
                expected_references_per_image=expected,
            ),
            False,
        )
    if config.scst.reward == "cider_d":
        corpus = [[f"{caption} <eos>" for caption in record.captions] for record in train_records]
        return (
            CiderDReward(
                corpus,
                expected_references_per_image=expected,
            ),
            True,
        )
    raise ValueError(f"Unsupported SCST reward: {config.scst.reward}")


def _reward_protocol_metadata(config: ExperimentConfig) -> dict[str, Any]:
    package = "nltk" if config.scst.reward == "nltk_meteor" else "pycocoevalcap"
    try:
        package_version = version(package)
    except PackageNotFoundError:
        package_version = "unknown"
    if config.scst.reward == "nltk_meteor":
        return {
            "name": "nltk_meteor",
            "implementation": "nltk.translate.meteor_score.meteor_score",
            "package_version": package_version,
            "tokenization": "whitespace_casefold_tokens_v1",
            "eos_policy": "excluded",
        }
    return {
        "name": "cider_d",
        "implementation": "scst_captioner._cider.CiderDScorer",
        "document_frequency": "fixed_training_corpus_v2",
        "package_version": package_version,
        "tokenization": CIDER_D_TOKENIZATION,
        "eos_policy": "append_on_completion",
    }


def command_train_scst(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    records, manifest = _load_corpus(config)
    tokenizer, processor = _load_hf_components(config)
    train_records = records_for_split(records, manifest, "train", validate=False)
    validation_records = records_for_split(records, manifest, "validation", validate=False)
    reward, include_eos = _make_reward(config, train_records)
    train_dataset = SCSTImageDataset(
        train_records,
        image_processor=processor,
        pil_transform=_train_augmentation(config),
        expected_references_per_image=config.data.expected_references_per_image,
    )
    validation_dataset = SCSTImageDataset(
        validation_records,
        image_processor=processor,
        expected_references_per_image=config.data.expected_references_per_image,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=config.scst.batch_size,
        shuffle=True,
        config=config,
        collate_fn=collate_image_references,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=config.scst.batch_size,
        shuffle=False,
        config=config,
        collate_fn=collate_image_references,
    )
    model = build_model(config.model, tokenizer).to(device)
    source_schema = None
    if args.resume is None:
        if args.xe_checkpoint is None:
            raise ValueError("--xe-checkpoint is required unless --resume is supplied")
        source_metadata = load_model_checkpoint(model, args.xe_checkpoint, map_location="cpu")
        _validate_checkpoint_config(
            source_metadata,
            config,
            expected_stage="xe",
            allow_legacy=args.allow_unverified_legacy_xe,
        )
        if not (source_metadata.get("legacy") or source_metadata.get("unverified_legacy")):
            _validate_checkpoint_data(source_metadata, config, manifest)
            _validate_checkpoint_scientific_protocol(source_metadata, config)
        source_schema = source_metadata.get("schema_version", 0)
        del source_metadata
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.scst.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scst.lr_decay)
    trainer = SCSTTrainer(
        model,
        optimizer,
        tokenizer=tokenizer,
        reward=reward,
        device=device,
        max_new_tokens=config.model.max_seq_length,
        sample_model_mode=config.scst.sample_model_mode,
        include_eos_in_reward=include_eos,
        scheduler=scheduler,
        amp=config.runtime.amp,
        gradient_clip_norm=config.scst.gradient_clip_norm,
    )
    start_epoch, global_step, resume_checkpoint = _resume_if_requested(
        args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=trainer.scaler,
        config=config,
        manifest=manifest,
        expected_stage="scst",
    )
    _restore_loader_generator(
        train_loader,
        (resume_checkpoint or {}).get("training_state", {}).get("data_loader_generator_state"),
    )
    trainer.global_step = global_step
    run_dir = prepare_run_dir(
        args.output_dir or Path(config.runtime.output_dir) / "scst",
        overwrite=args.overwrite or args.resume is not None,
    )
    provenance = _run_provenance(
        config,
        manifest=manifest,
        extra={
            "stage": "scst",
            "xe_checkpoint": (
                str(Path(args.xe_checkpoint))
                if args.resume is None and args.xe_checkpoint is not None
                else None
            ),
            "xe_checkpoint_sha256": (
                sha256_file(args.xe_checkpoint)
                if args.resume is None and args.xe_checkpoint is not None
                else None
            ),
            "xe_checkpoint_schema": source_schema,
            "resume_checkpoint": str(Path(args.resume)) if args.resume else None,
            "resume_checkpoint_sha256": sha256_file(args.resume) if args.resume else None,
            "parent_provenance": (resume_checkpoint or {}).get("provenance", {}),
            "reward": config.scst.reward,
            "reward_protocol": _reward_protocol_metadata(config),
            "reward_protocol_sha256": _sha256_json(_reward_protocol_metadata(config)),
            "reward_eos_policy": "append_on_completion" if include_eos else "excluded",
            "baseline": "greedy_eval_mode",
            "sample_model_mode": config.scst.sample_model_mode,
            "rollout_inference_precision": (
                "cuda_amp_fp16" if config.runtime.amp and device.type == "cuda" else "fp32"
            ),
        },
    )
    _write_run_metadata(run_dir, config, provenance)

    history_path = run_dir / "history.json"
    history: list[dict[str, Any]] = (
        _resume_history(
            history_path,
            resume_checkpoint,
            expected_last_epoch=start_epoch - 1,
        )
        if args.resume
        else []
    )
    best_reward = _best_metric_value(resume_checkpoint, -float("inf"))
    for epoch in range(start_epoch, config.scst.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        validation_reward = trainer.evaluate(validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train": dataclasses.asdict(train_metrics),
                "validation_reward": validation_reward,
            }
        )
        improved = validation_reward > best_reward
        if improved:
            best_reward = validation_reward
        checkpoint = build_checkpoint(
            model,
            stage="scst",
            config=config.to_dict(),
            epoch=epoch,
            global_step=trainer.global_step,
            best_metric={
                "name": f"validation_{config.scst.reward}",
                "value": best_reward,
                "mode": "max",
            },
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            provenance=provenance,
            training_state={
                "history": history,
                "data_loader_generator_state": (
                    train_loader.generator.get_state()
                    if train_loader.generator is not None
                    else None
                ),
            },
        )
        save_checkpoint(checkpoint, run_dir / "last.pt")
        if improved:
            save_checkpoint(checkpoint, run_dir / "best.pt")
        write_json(run_dir / "history.json", history)
        print(
            f"SCST epoch {epoch}: loss={train_metrics.loss:.4f}, "
            f"sample={train_metrics.sampled_reward:.4f}, baseline={train_metrics.baseline_reward:.4f}, "
            f"validation={validation_reward:.4f}"
        )
    return 0


@torch.inference_mode()
def _generate_records(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    tokenizer: Any,
    device: torch.device,
    config: ExperimentConfig,
    decoding: str,
) -> tuple[list[PredictionRecord], list[ReferenceRecord], list[bool]]:
    special = resolve_special_token_ids(tokenizer)
    previous_mode = model.training
    model.eval()
    predictions: list[PredictionRecord] = []
    references: list[ReferenceRecord] = []
    completions: list[bool] = []
    try:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            amp_enabled = bool(config.runtime.amp and device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                memory = model.encode_images(pixel_values)
                if decoding == "greedy":
                    output = greedy_decode(
                        model,
                        memory,
                        bos_token_id=special.bos,
                        eos_token_id=special.eos,
                        max_new_tokens=config.model.max_seq_length,
                        forbidden_token_ids=special.forbidden,
                    )
                elif decoding == "beam":
                    output = beam_decode(
                        model,
                        memory,
                        bos_token_id=special.bos,
                        eos_token_id=special.eos,
                        max_new_tokens=config.model.max_seq_length,
                        beam_size=config.evaluation.beam_size,
                        length_penalty_alpha=config.evaluation.length_penalty_alpha,
                        forbidden_token_ids=special.forbidden,
                    )
                else:
                    raise ValueError(f"Unknown decoding method: {decoding}")
            captions = tokenizer.batch_decode(output.token_ids.cpu(), skip_special_tokens=True)
            for index, (image_id, caption, image_references) in enumerate(
                zip(batch["image_ids"], captions, batch["references"], strict=True)
            ):
                predictions.append(PredictionRecord(image_id, caption))
                references.append(ReferenceRecord(image_id, tuple(image_references)))
                active_actions = output.action_token_ids[index][output.action_mask[index]]
                completions.append(
                    bool(active_actions.numel() and int(active_actions[-1].item()) == special.eos)
                )
    finally:
        model.train(previous_mode)
    return predictions, references, completions


def _evaluation_setup(config: ExperimentConfig) -> tuple[Any, ...]:
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    records, manifest = _load_corpus(config)
    tokenizer, processor = _load_hf_components(config)
    train_records = records_for_split(records, manifest, "train", validate=False)
    test_records = records_for_split(records, manifest, "test", validate=False)
    dataset = SCSTImageDataset(
        test_records,
        image_processor=processor,
        expected_references_per_image=config.data.expected_references_per_image,
    )
    loader = _loader(
        dataset,
        batch_size=config.scst.batch_size,
        shuffle=False,
        config=config,
        collate_fn=collate_image_references,
    )
    model = build_model(config.model, tokenizer).to(device)
    return device, manifest, tokenizer, train_records, test_records, loader, model


def _evaluate_checkpoint(
    *,
    checkpoint_path: str,
    decoding: str,
    config: ExperimentConfig,
    device: torch.device,
    manifest: Any,
    tokenizer: Any,
    train_records: Sequence[Any],
    test_records: Sequence[Any],
    loader: DataLoader[Any],
    model: torch.nn.Module,
    expected_stage: str | None = None,
    allow_legacy: bool = False,
) -> tuple[Any, list[PredictionRecord], dict[str, Any], dict[str, Any]]:
    checkpoint = load_model_checkpoint(model, checkpoint_path, map_location="cpu")
    _validate_checkpoint_config(
        checkpoint,
        config,
        expected_stage=expected_stage,
        allow_legacy=allow_legacy,
    )
    if not (checkpoint.get("legacy") or checkpoint.get("unverified_legacy")):
        _validate_checkpoint_data(checkpoint, config, manifest)
        _validate_checkpoint_scientific_protocol(checkpoint, config)
    summary = {
        "stage": checkpoint.get("stage"),
        "legacy": bool(checkpoint.get("legacy")),
        "unverified_legacy": bool(checkpoint.get("unverified_legacy")),
        "config": checkpoint.get("config", {}),
        "provenance": checkpoint.get("provenance", {}),
    }
    del checkpoint
    predictions, references, completions = _generate_records(
        model,
        loader,
        tokenizer=tokenizer,
        device=device,
        config=config,
        decoding=decoding,
    )
    reward, include_eos = _make_reward(config, train_records)
    objective_predictions = [record.caption for record in predictions]
    objective_references = [list(record.captions) for record in references]
    if include_eos:
        objective_predictions = [
            f"{caption} <eos>".strip() if completed else caption
            for caption, completed in zip(objective_predictions, completions, strict=True)
        ]
        objective_references = [
            [f"{caption} <eos>".strip() for caption in image_references]
            for image_references in objective_references
        ]
    objective_values = tuple(
        float(value) for value in reward(objective_predictions, objective_references)
    )
    objective_result = {
        "name": config.scst.reward,
        "mean": sum(objective_values) / len(objective_values),
        "per_image": [
            {
                "image_id": record.image_id,
                "score": score,
                "completed": completed,
            }
            for record, score, completed in zip(
                predictions, objective_values, completions, strict=True
            )
        ],
        "eos_policy": "append_on_completion" if include_eos else "excluded",
        "protocol": _reward_protocol_metadata(config),
    }
    context = {
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest_sha256": sha256_file(config.data.split_manifest),
        "caption_records_sha256": manifest.records_sha256,
        "image_files_sha256": manifest.images_sha256,
        "decoding": decoding,
        "beam_size": config.evaluation.beam_size if decoding == "beam" else None,
        "length_penalty_alpha": (
            config.evaluation.length_penalty_alpha if decoding == "beam" else None
        ),
        "max_new_tokens": config.model.max_seq_length,
        "training_objective": _reward_protocol_metadata(config),
        "training_objective_eos_policy": objective_result["eos_policy"],
        "tokenizer_name": config.model.tokenizer_name,
        "tokenizer_revision": config.model.tokenizer_revision,
        "encoder_name": config.model.encoder_name,
        "encoder_revision": config.model.encoder_revision,
        "special_token_policy": "forbid_all_special_except_eos",
        "inference_precision": (
            "cuda_amp_fp16" if config.runtime.amp and device.type == "cuda" else "fp32"
        ),
        "unverified_legacy_checkpoint": bool(summary["legacy"] or summary["unverified_legacy"]),
    }
    result = evaluate_coco_caption_records(
        [record.image_id for record in test_records],
        predictions,
        references,
        expected_references_per_image=config.data.expected_references_per_image,
        include_spice=config.evaluation.include_spice,
        context=context,
    )
    return result, predictions, summary, objective_result


def _prediction_payload(
    predictions: Sequence[PredictionRecord], objective_result: dict[str, Any]
) -> list[dict[str, Any]]:
    objective_by_id = {item["image_id"]: item for item in objective_result.get("per_image", [])}
    return [
        {
            "image_id": record.image_id,
            "caption": record.caption,
            "completed": bool(objective_by_id[record.image_id]["completed"]),
        }
        for record in predictions
    ]


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device, manifest, tokenizer, train_records, test_records, loader, model = _evaluation_setup(
        config
    )
    result, predictions, summary, objective_result = _evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        decoding=args.decoding,
        config=config,
        device=device,
        manifest=manifest,
        tokenizer=tokenizer,
        train_records=train_records,
        test_records=test_records,
        loader=loader,
        model=model,
        allow_legacy=args.allow_unverified_legacy,
    )
    output_dir = prepare_run_dir(args.output_dir, overwrite=args.overwrite)
    result_payload = result.to_dict()
    result_payload["training_objective"] = objective_result
    result_payload["checkpoint_metadata"] = summary
    result_payload["resolved_config"] = config.to_dict()
    result_payload["environment"] = collect_environment(Path.cwd())
    predictions_path = output_dir / "predictions.json"
    write_json(predictions_path, _prediction_payload(predictions, objective_result))
    result_payload["prediction_artifact"] = {
        "path": predictions_path.name,
        "sha256": sha256_file(predictions_path),
    }
    write_json(output_dir / "metrics.json", result_payload)
    print({**result.metrics, config.scst.reward: objective_result["mean"]})
    return 0


def command_evaluate_matrix(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device, manifest, tokenizer, train_records, test_records, loader, model = _evaluation_setup(
        config
    )
    preflight_summaries = {
        "xe": _preflight_checkpoint(
            args.xe_checkpoint,
            expected_stage="xe",
            config=config,
            manifest=manifest,
        ),
        "scst": _preflight_checkpoint(
            args.scst_checkpoint,
            expected_stage="scst",
            config=config,
            manifest=manifest,
        ),
    }
    xe_checkpoint_sha256 = sha256_file(args.xe_checkpoint)
    recorded_parent = _recorded_xe_parent_sha256(preflight_summaries["scst"].get("provenance"))
    if recorded_parent != xe_checkpoint_sha256:
        raise RuntimeError(
            "SCST checkpoint lineage does not identify the supplied XE checkpoint as its parent"
        )
    output_dir = prepare_run_dir(args.output_dir, overwrite=args.overwrite)
    cells: dict[str, dict[str, Any]] = {}
    checkpoint_summaries: dict[str, dict[str, Any]] = {}
    protocol_hashes: set[str] = set()
    for stage, checkpoint_path in (("xe", args.xe_checkpoint), ("scst", args.scst_checkpoint)):
        cells[stage] = {}
        for decoding in ("greedy", "beam"):
            result, predictions, summary, objective_result = _evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                decoding=decoding,
                config=config,
                device=device,
                manifest=manifest,
                tokenizer=tokenizer,
                train_records=train_records,
                test_records=test_records,
                loader=loader,
                model=model,
                expected_stage=stage,
            )
            protocol_hashes.add(str(result.provenance["comparison_protocol_sha256"]))
            checkpoint_summaries[stage] = summary
            cell_payload = result.to_dict()
            cell_payload["training_objective"] = objective_result
            cell_payload["checkpoint_metadata"] = summary
            predictions_path = output_dir / f"{stage}_{decoding}_predictions.json"
            write_json(
                predictions_path,
                _prediction_payload(predictions, objective_result),
            )
            cell_payload["prediction_artifact"] = {
                "path": predictions_path.name,
                "sha256": sha256_file(predictions_path),
            }
            cells[stage][decoding] = cell_payload
    if len(protocol_hashes) != 1:
        raise RuntimeError("Evaluation cells did not share one comparison protocol")
    exemplar_provenance = cells["xe"]["greedy"]["provenance"]
    full_protocol = {
        "schema_version": 1,
        "split_manifest_sha256": sha256_file(config.data.split_manifest),
        "caption_records_sha256": manifest.records_sha256,
        "image_files_sha256": manifest.images_sha256,
        "ordered_image_ids_sha256": exemplar_provenance["ordered_image_ids_sha256"],
        "references_sha256": exemplar_provenance["references_sha256"],
        "metric_backend": exemplar_provenance["backend"],
        "metric_backend_version": exemplar_provenance["backend_version"],
        "metric_names": exemplar_provenance["metrics"],
        "tokenizer_name": config.model.tokenizer_name,
        "tokenizer_revision": config.model.tokenizer_revision,
        "image_processor_name": config.model.encoder_name,
        "image_processor_revision": config.model.encoder_revision,
        "max_new_tokens": config.model.max_seq_length,
        "special_token_policy": "forbid_all_special_except_eos",
        "inference_precision": (
            "cuda_amp_fp16" if config.runtime.amp and device.type == "cuda" else "fp32"
        ),
        "training_objective": _reward_protocol_metadata(config),
        "sample_model_mode": config.scst.sample_model_mode,
        "baseline": "greedy_eval_mode",
        "decoders": {
            "greedy": {},
            "beam": {
                "beam_size": config.evaluation.beam_size,
                "length_penalty_alpha": config.evaluation.length_penalty_alpha,
            },
        },
        "evaluation_preprocessing": "deterministic_huggingface_image_processor",
    }
    matrix = {
        "metric_corpus_protocol_sha256": next(iter(protocol_hashes)),
        "full_protocol_sha256": _sha256_json(full_protocol),
        "full_protocol": full_protocol,
        "split_manifest_sha256": sha256_file(config.data.split_manifest),
        "checkpoints": {
            "xe": {"path": args.xe_checkpoint, "sha256": xe_checkpoint_sha256},
            "scst": {
                "path": args.scst_checkpoint,
                "sha256": sha256_file(args.scst_checkpoint),
            },
        },
        "checkpoint_metadata": checkpoint_summaries,
        "resolved_config": config.to_dict(),
        "environment": collect_environment(Path.cwd()),
        "cells": cells,
    }
    write_json(output_dir / "matrix.json", matrix)
    print(f"Wrote the XE/SCST x greedy/beam matrix to {output_dir / 'matrix.json'}")
    return 0


def command_caption(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = resolve_device(config.runtime.device)
    tokenizer, processor = _load_hf_components(config)
    model = build_model(config.model, tokenizer).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, map_location="cpu")
    _validate_checkpoint_config(checkpoint, config, allow_legacy=True)
    del checkpoint
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    special = resolve_special_token_ids(tokenizer)
    model.eval()
    with torch.inference_mode():
        amp_enabled = bool(config.runtime.amp and device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            memory = model.encode_images(pixel_values)
            if args.decoding == "beam":
                output = beam_decode(
                    model,
                    memory,
                    bos_token_id=special.bos,
                    eos_token_id=special.eos,
                    max_new_tokens=config.model.max_seq_length,
                    beam_size=config.evaluation.beam_size,
                    length_penalty_alpha=config.evaluation.length_penalty_alpha,
                    forbidden_token_ids=special.forbidden,
                )
            else:
                output = greedy_decode(
                    model,
                    memory,
                    bos_token_id=special.bos,
                    eos_token_id=special.eos,
                    max_new_tokens=config.model.max_seq_length,
                    forbidden_token_ids=special.forbidden,
                )
    print(tokenizer.decode(output.token_ids[0].cpu(), skip_special_tokens=True))
    return 0


def command_convert_checkpoint(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tokenizer, _ = _load_hf_components(config)
    model = build_model(config.model, tokenizer)
    metadata = load_model_checkpoint(model, args.input, map_location="cpu", strict=True)
    if not metadata.get("legacy"):
        raise ValueError("Input is already a versioned checkpoint")
    provenance = {
        "lineage": "legacy_notebook_raw_state_dict",
        "source_path": str(Path(args.input)),
        "source_sha256": sha256_file(args.input),
        "unknown_fields": [
            "epoch",
            "optimizer_state",
            "scheduler_state",
            "rng_state",
            "split_manifest",
            "code_revision",
        ],
    }
    checkpoint = build_checkpoint(
        model,
        stage=args.stage,
        config=config.to_dict(),
        provenance=provenance,
        include_rng_state=False,
    )
    checkpoint["unverified_legacy"] = True
    save_checkpoint(checkpoint, args.output)
    print(f"Converted strict legacy weights to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scst-captioner",
        description="Reproducible CPTR-inspired image captioning with SCST",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Create a persisted image split")
    prepare.add_argument("--config", default="configs/default.toml")
    prepare.add_argument("--official-train")
    prepare.add_argument("--official-validation")
    prepare.add_argument("--official-test")
    prepare.add_argument("--train-fraction", type=float, default=0.8)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=command_prepare_data)

    validate = subparsers.add_parser("validate-data", help="Validate records and split")
    validate.add_argument("--config", default="configs/default.toml")
    validate.set_defaults(handler=command_validate_data)

    train_xe = subparsers.add_parser("train-xe", help="Run teacher-forced XE pretraining")
    train_xe.add_argument("--config", default="configs/default.toml")
    train_xe.add_argument("--output-dir")
    train_xe.add_argument("--resume")
    train_xe.add_argument("--overwrite", action="store_true")
    train_xe.set_defaults(handler=command_train_xe)

    train_scst = subparsers.add_parser("train-scst", help="Fine-tune an XE model with SCST")
    train_scst.add_argument("--config", default="configs/default.toml")
    train_scst.add_argument("--xe-checkpoint")
    train_scst.add_argument("--output-dir")
    train_scst.add_argument("--resume")
    train_scst.add_argument("--allow-unverified-legacy-xe", action="store_true")
    train_scst.add_argument("--overwrite", action="store_true")
    train_scst.set_defaults(handler=command_train_scst)

    evaluate = subparsers.add_parser("evaluate", help="Run one COCO-protocol evaluation")
    evaluate.add_argument("--config", default="configs/default.toml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--decoding", choices=("greedy", "beam"), default="greedy")
    evaluate.add_argument("--output-dir", default="artifacts/evaluation")
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument("--allow-unverified-legacy", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)

    matrix = subparsers.add_parser(
        "evaluate-matrix", help="Run the XE/SCST by greedy/beam comparison"
    )
    matrix.add_argument("--config", default="configs/default.toml")
    matrix.add_argument("--xe-checkpoint", required=True)
    matrix.add_argument("--scst-checkpoint", required=True)
    matrix.add_argument("--output-dir", default="artifacts/evaluation-matrix")
    matrix.add_argument("--overwrite", action="store_true")
    matrix.set_defaults(handler=command_evaluate_matrix)

    caption = subparsers.add_parser("caption", help="Caption one local image")
    caption.add_argument("image")
    caption.add_argument("--config", default="configs/default.toml")
    caption.add_argument("--checkpoint", required=True)
    caption.add_argument("--decoding", choices=("greedy", "beam"), default="beam")
    caption.set_defaults(handler=command_caption)

    convert = subparsers.add_parser(
        "convert-checkpoint", help="Wrap a strict legacy raw state dict"
    )
    convert.add_argument("--config", default="configs/default.toml")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--stage", choices=("xe", "scst", "unknown"), default="unknown")
    convert.set_defaults(handler=command_convert_checkpoint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
