"""Compare both decoder paths on an existing XE checkpoint and validation images.

Run from the repository root: python scripts/verify_kv_cache.py
Uses local pretrained assets only and never modifies the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, ViTConfig, ViTModel

from scst_captioner.checkpointing import load_checkpoint_file
from scst_captioner.config import ModelConfig
from scst_captioner.generation import greedy_decode, sample_decode
from scst_captioner.losses import reinforce_loss
from scst_captioner.model import CPTRCaptioner
from scst_captioner.tokenization import decode_caption_batch, resolve_special_token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/xe/best.pt")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--output", default="artifacts/kv_cache_verification.json")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint_file(args.checkpoint)
    config = ModelConfig(**checkpoint["config"]["model"])
    data = checkpoint["config"]["data"]
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, revision=config.tokenizer_revision, local_files_only=True
    )
    processor = AutoImageProcessor.from_pretrained(
        config.encoder_name, revision=config.encoder_revision, local_files_only=True
    )
    backbone = ViTModel(
        ViTConfig.from_pretrained(
            config.encoder_name, revision=config.encoder_revision, local_files_only=True
        ),
        add_pooling_layer=False,
    )
    model = CPTRCaptioner(
        config, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, backbone=backbone
    ).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    special = resolve_special_token_ids(tokenizer)
    manifest = json.loads(Path(data["split_manifest"]).read_text())
    validation = manifest["splits"].get("validation", manifest["splits"].get("val"))
    image_ids = validation[: args.images]
    images = []
    for name in image_ids:
        with Image.open(Path(data["images_dir"]) / name) as image:
            images.append(image.convert("RGB"))
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        memory = model.encode_images(pixels)
    kwargs = dict(
        bos_token_id=special.bos,
        eos_token_id=special.eos,
        max_new_tokens=config.max_seq_length,
        forbidden_token_ids=special.forbidden,
    )
    report = {
        "checkpoint": args.checkpoint,
        "device": "cpu",
        "dtype": "float32",
        "images": image_ids,
        "strict_checkpoint_load": True,
    }
    # Warm both implementations before timing the same batch.
    with torch.no_grad():
        for cached in (False, True):
            greedy_decode(model, memory[:1], **{**kwargs, "max_new_tokens": 3}, use_kv_cache=cached)
        outputs = []
        for cached in (False, True):
            start = time.perf_counter()
            output = greedy_decode(model, memory, **kwargs, use_kv_cache=cached)
            elapsed = time.perf_counter() - start
            report["cached_seconds" if cached else "uncached_seconds"] = elapsed
            outputs.append(output)
            print(f"greedy cache={cached}: {elapsed:.3f}s", flush=True)
        assert torch.equal(outputs[0].token_ids, outputs[1].token_ids)
        assert torch.equal(outputs[0].action_mask, outputs[1].action_mask)
        report["greedy_tokens_identical"] = True
        report["captions"] = decode_caption_batch(tokenizer, outputs[0].token_ids)
        samples = []
        for cached in (False, True):
            torch.manual_seed(123)
            samples.append(sample_decode(model, memory[:2], **kwargs, use_kv_cache=cached))
        report["seeded_sample_tokens_identical"] = torch.equal(
            samples[0].token_ids, samples[1].token_ids
        )
        # Freeze sampled actions to compare the exact same SCST objective.
        actions = samples[0].action_token_ids
        mask = samples[0].action_mask
        prefixes = samples[0].token_ids[:, :-1]
    reference = None
    for cached in (False, True):
        model.zero_grad(set_to_none=True)
        inputs = memory[:2].detach().requires_grad_(True)
        cache = model.init_decode_cache(inputs) if cached else None
        scores = []
        logits_list = []
        for step in range(prefixes.size(1)):
            if cached:
                logits, cache = model.decode_step(prefixes[:, step : step + 1], cache)
            else:
                logits = model.predict_next_token(prefixes[:, : step + 1], inputs)
            logits_list.append(logits.detach())
            logits = logits.clone()
            logits[:, list(special.forbidden)] = -torch.inf
            scores.append(logits.log_softmax(-1).gather(1, actions[:, step : step + 1]).squeeze(1))
        scores = torch.stack(scores, 1)
        loss = reinforce_loss(torch.tensor([1.25, -0.75]), scores, mask)
        loss.backward()
        result = (
            torch.stack(logits_list, 1),
            scores.detach(),
            loss.detach(),
            inputs.grad.clone(),
            {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None},
        )
        print(f"gradient comparison cache={cached}: loss={loss.item():.8f}", flush=True)
        if not cached:
            reference = result
        else:
            for left, right in zip(reference[:4], result[:4], strict=True):
                torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)
            assert reference[4].keys() == result[4].keys()
            max_grad_error = 0.0
            for name in reference[4]:
                torch.testing.assert_close(
                    reference[4][name],
                    result[4][name],
                    atol=3e-5,
                    rtol=3e-4,
                    msg=lambda msg, name=name: f"{name}: {msg}",
                )
                max_grad_error = max(
                    max_grad_error, (reference[4][name] - result[4][name]).abs().max().item()
                )
            report.update(
                max_logit_abs_error=(reference[0] - result[0]).abs().max().item(),
                loss_abs_error=(reference[2] - result[2]).abs().item(),
                max_parameter_gradient_abs_error=max_grad_error,
                gradient_parameters_checked=len(result[4]),
                sampled_steps=prefixes.size(1),
                gradient_comparison_passed=True,
            )
    report["greedy_speedup"] = report["uncached_seconds"] / report["cached_seconds"]
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("images", "captions")}, indent=2)
    )


if __name__ == "__main__":
    main()
