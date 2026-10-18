"""Compare both decoder paths on an existing XE checkpoint and validation images.

Run from the repository root: python scripts/verify_kv_cache.py
Uses local pretrained assets only and never modifies the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, ViTConfig, ViTModel

from scst_captioner.checkpointing import load_checkpoint_file, sha256_file
from scst_captioner.config import ModelConfig
from scst_captioner.generation import beam_decode, greedy_decode, sample_decode
from scst_captioner.losses import reinforce_loss
from scst_captioner.model import CPTRCaptioner
from scst_captioner.tokenization import decode_caption_batch, resolve_special_token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/xe/best.pt")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--output", default="artifacts/kv_cache_verification.json")
    parser.add_argument("--decoding", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.images < 1:
        parser.error("--images must be positive")
    if args.amp and args.device != "cuda":
        parser.error("--amp requires CUDA")
    if args.decoding != "beam" and (args.device != "cpu" or args.amp):
        parser.error("device/precision options currently apply to beam verification")
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
    if args.decoding == "beam":
        verify_beam(model, tokenizer, pixels, config, image_ids, args)
        return
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


def verify_beam(model, tokenizer, pixels, config, image_ids, args):
    """Compare immutable beam branches on saved weights and fixed validation images."""
    device = torch.device(args.device)
    model.to(device)
    special = resolve_special_token_ids(tokenizer)
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "dtype": "float16" if args.amp else "float32",
        "beam_size": 3,
        "length_penalty_alpha": 0.7,
        "images": image_ids,
        "strict_checkpoint_load": True,
        "comparisons": [],
    }
    kwargs = dict(
        bos_token_id=special.bos,
        eos_token_id=special.eos,
        max_new_tokens=config.max_seq_length,
        forbidden_token_ids=special.forbidden,
        beam_size=3,
        length_penalty_alpha=0.7,
    )
    failures = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext(),
    ):
        for index, image_id in enumerate(image_ids):
            memory = model.encode_images(pixels[index : index + 1].to(device))
            if index == 0:
                for cached in (False, True):
                    beam_decode(
                        model, memory, **{**kwargs, "max_new_tokens": 3}, use_kv_cache=cached
                    )
            outputs = []
            row = {"image_id": image_id}
            for cached in (False, True):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sync()
                start = time.perf_counter()
                output = beam_decode(model, memory, **kwargs, use_kv_cache=cached)
                sync()
                label = "cached" if cached else "uncached"
                row[label + "_seconds"] = time.perf_counter() - start
                row[label + "_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else None
                )
                outputs.append(output)
            row["tokens_identical"] = torch.equal(outputs[0].token_ids, outputs[1].token_ids)
            row["masks_identical"] = torch.equal(outputs[0].action_mask, outputs[1].action_mask)
            row["raw_score_abs_error"] = (
                (outputs[0].sequence_log_probs - outputs[1].sequence_log_probs).abs().max().item()
            )
            row["captions"] = [
                decode_caption_batch(tokenizer, out.token_ids.cpu())[0] for out in outputs
            ]
            # Check logits along the winning full-prefix path, not only the final text.
            cache = model.init_decode_cache(memory)
            prefix = outputs[0].token_ids[:, :-1]
            max_error = 0.0
            logits_close = True
            for step in range(prefix.size(1)):
                full = model.predict_next_token(prefix[:, : step + 1], memory)
                incremental, cache = model.decode_step(prefix[:, step : step + 1], cache)
                max_error = max(max_error, (full - incremental).abs().max().item())
                logits_close &= torch.allclose(
                    full,
                    incremental,
                    atol=2e-2 if args.amp else 1e-4,
                    rtol=2e-3 if args.amp else 1e-4,
                )
            row["max_logit_abs_error"] = max_error
            row["logits_close"] = logits_close
            row["scores_close"] = torch.allclose(
                outputs[0].sequence_log_probs,
                outputs[1].sequence_log_probs,
                atol=0.1 if args.amp else 1e-3,
                rtol=1e-4,
            )
            row["passed"] = all(
                row[key]
                for key in ("tokens_identical", "masks_identical", "scores_close", "logits_close")
            )
            if not row["passed"]:
                failures.append(image_id)
            report["comparisons"].append(row)
            print(
                f"beam {index + 1}/{len(image_ids)} {image_id}: passed={row['passed']} "
                f"uncached={row['uncached_seconds']:.2f}s cached={row['cached_seconds']:.2f}s",
                flush=True,
            )
    report["passed"] = not failures
    report["failed_images"] = failures
    report["speedup"] = sum(r["uncached_seconds"] for r in report["comparisons"]) / sum(
        r["cached_seconds"] for r in report["comparisons"]
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"Beam cache verification failed for {failures}; see {destination}")
    print(f"Beam verification passed; decoder-only speedup {report['speedup']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
