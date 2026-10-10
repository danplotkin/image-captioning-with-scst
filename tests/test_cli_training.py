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
        assert uninterrupted["progress"]["global_step"] > 0
        assert type(uninterrupted["provenance"]["environment"]["torch"]) is str
        assert (directory / "best.pt").is_file()

        args.resume = str(directory / "epoch1.pt")
        args.output_dir = str(tmp_path / f"{stage}-resumed")
        assert command(args) == 0
        resumed = load_checkpoint_file(Path(args.output_dir) / "last.pt")
        assert resumed["progress"] == uninterrupted["progress"]
        assert resumed["training_state"]["history"] == uninterrupted["training_state"]["history"]
        for name, tensor in uninterrupted["model_state"].items():
            assert torch.equal(tensor, resumed["model_state"][name]), name
