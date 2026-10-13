from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import BertTokenizerFast, ViTConfig, ViTImageProcessor, ViTModel

from scst_captioner import cli
from scst_captioner.checkpointing import load_checkpoint_file, save_checkpoint
from scst_captioner.config import ExperimentConfig, ModelConfig
from scst_captioner.model import CPTRCaptioner


def test_cli_xe_to_scst_and_epoch_resume_with_real_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pycocoevalcap")
    images = tmp_path / "Images"
    images.mkdir()
    captions_file = tmp_path / "captions.txt"
    with captions_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "caption"])
        for index, (color, animal) in enumerate(
            [("red", "cat"), ("blue", "dog"), ("green", "bird"), ("white", "horse")]
        ):
            image_id = f"{index}.jpg"
            Image.new("RGB", (20, 18), color).save(images / image_id)
            writer.writerows(
                (image_id, caption)
                for caption in (
                    f"a {color} {animal}",
                    f"a {color} {animal}",
                    f"the {animal} runs",
                    f"a small {animal} is outside",
                    f"the {color} animal runs",
                )
            )
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "a",
                "the",
                "red",
                "blue",
                "green",
                "white",
                "cat",
                "dog",
                "bird",
                "horse",
                "animal",
                "runs",
                "small",
                "is",
                "outside",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)
    processor = ViTImageProcessor(size={"height": 16, "width": 16})
    base = ExperimentConfig()
    config = dataclasses.replace(
        base,
        model=dataclasses.replace(
            base.model,
            d_model=8,
            num_heads=2,
            feedforward_dim=16,
            num_decoder_layers=1,
            max_seq_length=8,
        ),
        data=dataclasses.replace(
            base.data,
            images_dir=str(images),
            captions_file=str(captions_file),
            split_manifest=str(tmp_path / "split.json"),
            num_workers=0,
            horizontal_flip_probability=0.0,
            rotation_degrees=0.0,
            color_jitter_probability=0.0,
        ),
        xe=dataclasses.replace(base.xe, epochs=2, batch_size=5, patience=3, warmup_ratio=0.0),
        scst=dataclasses.replace(base.scst, epochs=2, batch_size=2),
        runtime=dataclasses.replace(base.runtime, device="cpu", amp=False),
    )

    def tiny_model(model_config: ModelConfig, current_tokenizer: BertTokenizerFast):
        backbone = ViTModel(
            ViTConfig(
                image_size=16,
                patch_size=4,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            ),
            add_pooling_layer=False,
        )
        return CPTRCaptioner(
            model_config,
            vocab_size=len(current_tokenizer),
            pad_token_id=current_tokenizer.pad_token_id,
            backbone=backbone,
        )

    def save_epoch_snapshot(checkpoint, path):
        save_checkpoint(checkpoint, path)
        if Path(path).name == "last.pt" and checkpoint["progress"]["epoch"] == 1:
            save_checkpoint(checkpoint, Path(path).with_name("epoch1.pt"))

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "_load_hf_components", lambda _: (tokenizer, processor))
    monkeypatch.setattr(cli, "build_model", tiny_model)
    monkeypatch.setattr(cli, "save_checkpoint", save_epoch_snapshot)
    validation_scores = []
    original_evaluate = cli.SCSTTrainer.evaluate

    def record_validation(trainer, loader):
        score = original_evaluate(trainer, loader)
        validation_scores.append(score)
        return score

    monkeypatch.setattr(cli.SCSTTrainer, "evaluate", record_validation)
    # Actual corpus loading, preprocessing, model/trainer, reward, provenance,
    # checkpoint serialization and resume execute without metadata workarounds.
    assert (
        cli.command_prepare_data(
            argparse.Namespace(
                config="fixture.toml",
                official_train=None,
                official_validation=None,
                official_test=None,
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                overwrite=False,
            )
        )
        == 0
    )

    for stage, command in (("xe", cli.command_train_xe), ("scst", cli.command_train_scst)):
        directory = tmp_path / stage
        args = argparse.Namespace(
            config="fixture.toml",
            output_dir=str(directory),
            resume=None,
            overwrite=False,
            xe_checkpoint=str(tmp_path / "xe" / "best.pt"),
            allow_unverified_legacy_xe=False,
        )
        assert command(args) == 0
        uninterrupted = load_checkpoint_file(directory / "last.pt")
        if stage == "scst":
            assert len(validation_scores) == 3  # XE once, then each trained epoch.
            xe_score = validation_scores[0]
            assert uninterrupted["training_state"]["xe_validation_reward"] == xe_score
            for row in uninterrupted["training_state"]["history"]:
                assert row["xe_validation_reward"] == xe_score
                assert row["delta_vs_xe"] == row["validation_reward"] - xe_score
            with Image.open(directory / "reward_curve.png") as curve:
                curve.verify()
        assert uninterrupted["progress"]["global_step"] > 0
        assert type(uninterrupted["provenance"]["environment"]["torch"]) is str
        assert (directory / "best.pt").is_file()
        with Image.open(directory / "loss_curve.png") as curve:
            assert curve.format == "PNG"
            assert curve.width > 0 and curve.height > 0

        if stage == "xe":
            # Simulate a checkpoint written before optional fine-tuning fields.
            old_checkpoint = load_checkpoint_file(directory / "epoch1.pt")
            for key in ("encoder_trainable_blocks", "encoder_train_final_layer_norm"):
                old_checkpoint["config"]["model"].pop(key)
            for key in ("init_checkpoint", "encoder_learning_rate"):
                old_checkpoint["config"]["xe"].pop(key)
            save_checkpoint(old_checkpoint, directory / "epoch1.pt")
        args.resume = str(directory / "epoch1.pt")
        args.output_dir = str(tmp_path / f"{stage}-resumed")
        assert command(args) == 0
        resumed = load_checkpoint_file(Path(args.output_dir) / "last.pt")
        if stage == "scst":
            assert len(validation_scores) == 4  # Resume evaluates only the new epoch.
            assert resumed["training_state"]["xe_validation_reward"] == xe_score
            with Image.open(Path(args.output_dir) / "reward_curve.png") as curve:
                curve.verify()
        with Image.open(Path(args.output_dir) / "loss_curve.png") as curve:
            curve.verify()
        assert resumed["progress"] == uninterrupted["progress"]
        assert resumed["training_state"]["history"] == uninterrupted["training_state"]["history"]
        for name, tensor in uninterrupted["model_state"].items():
            assert torch.equal(tensor, resumed["model_state"][name]), name

    # A new XE run can change freezing policy and optimizer groups while keeping
    # the original data protocol. Its own epoch-boundary resume remains exact.
    parent_path = tmp_path / "xe" / "best.pt"
    parent = load_checkpoint_file(parent_path)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            freeze_encoder=False,
            encoder_trainable_blocks=1,
            encoder_train_final_layer_norm=True,
        ),
        xe=dataclasses.replace(
            config.xe,
            init_checkpoint=str(parent_path),
            learning_rate=1e-5,
            encoder_learning_rate=1e-6,
        ),
    )
    args = argparse.Namespace(
        config="fixture.toml",
        output_dir=str(tmp_path / "fine"),
        resume=None,
        overwrite=False,
    )
    assert cli.command_train_xe(args) == 0
    fine = load_checkpoint_file(tmp_path / "fine" / "last.pt")
    assert len(fine["training_state"]["history"]) == 2
    assert fine["progress"]["global_step"] == 4  # Two new epochs, two batches each.
    assert fine["provenance"]["init_checkpoint_sha256"] == cli.sha256_file(parent_path)
    assert len(fine["optimizer_state"]["param_groups"]) == 2
    embedding_key = "backbone.backbone.embeddings.patch_embeddings.projection.weight"
    block_key = "backbone.backbone.encoder.layer.0.output.dense.weight"
    assert torch.equal(parent["model_state"][embedding_key], fine["model_state"][embedding_key])
    assert not torch.equal(parent["model_state"][block_key], fine["model_state"][block_key])
    args.resume = str(tmp_path / "fine" / "epoch1.pt")
    args.output_dir = str(tmp_path / "fine-resumed")
    with pytest.raises(ValueError, match=r"Remove xe\.init_checkpoint"):
        cli.command_train_xe(args)
    config = dataclasses.replace(config, xe=dataclasses.replace(config.xe, init_checkpoint=None))
    assert cli.command_train_xe(args) == 0
    resumed = load_checkpoint_file(tmp_path / "fine-resumed" / "last.pt")
    assert resumed["progress"] == fine["progress"]
    assert resumed["training_state"]["history"] == fine["training_state"]["history"]
    for name, tensor in fine["model_state"].items():
        assert torch.equal(tensor, resumed["model_state"][name]), name
