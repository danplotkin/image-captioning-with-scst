"""Verify isolated AMP rollouts on the retained XE checkpoint without optimizer updates."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, ViTConfig, ViTModel

from scst_captioner.checkpointing import load_checkpoint_file, sha256_file
from scst_captioner.config import ModelConfig
from scst_captioner.generation import greedy_decode, sample_decode, scst_rollouts
from scst_captioner.model import CPTRCaptioner
from scst_captioner.tokenization import resolve_special_token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/flickr30k-xe-finetune-runpod/best.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("verification/amp-gradients.json"))
    parser.add_argument("--require-pod-runtime", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA; CPU tests cannot validate the pod runtime")
    if args.require_pod_runtime and (
        torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.8"
    ):
        raise RuntimeError(
            f"Expected PyTorch 2.8.0 / CUDA 12.8, got {torch.__version__}/{torch.version.cuda}"
        )
    torch.set_num_threads(4)
    saved = load_checkpoint_file(args.checkpoint)
    config = ModelConfig(**saved["config"]["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, revision=config.tokenizer_revision
    )
    processor = AutoImageProcessor.from_pretrained(
        config.encoder_name, revision=config.encoder_revision
    )
    backbone = ViTModel(
        ViTConfig.from_pretrained(config.encoder_name, revision=config.encoder_revision),
        add_pooling_layer=False,
    )
    model = CPTRCaptioner(
        config, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, backbone=backbone
    ).cuda()
    model.load_state_dict(saved["model_state"])
    special = resolve_special_token_ids(tokenizer)
    data = saved["config"]["data"]
    image_id = json.loads(Path(data["split_manifest"]).read_text())["splits"]["validation"][0]
    with Image.open(Path(data["images_dir"]) / image_id) as image:
        pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].cuda()
    options = dict(
        bos_token_id=special.bos,
        eos_token_id=special.eos,
        max_new_tokens=3,
        forbidden_token_ids=special.forbidden,
    )
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "validation_image": image_id,
        "trainable_parameter_tensors": sum(p.requires_grad for p in model.parameters()),
        "optimizer_updates": 0,
        "cases": [],
    }
    del saved
    # A positive control: recreate the original shared-context ordering directly.
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=True):
        memory = model.encode_images(pixels)
        with torch.no_grad():
            greedy_decode(model, memory.detach(), **options)
        old_sample = sample_decode(model, memory, **options)
    old_sample.action_log_probs.sum().backward()
    report["historical_shared_context_missing_gradients"] = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    del memory, old_sample
    original_autocast = torch.autocast
    for baseline in ("greedy", "none"):
        for mode in ("eval", "train"):
            for kv_cache in (True, False):
                reference = None
                for weight_cache in (False, True):
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(42)

                    def context(weight_cache=weight_cache, **kwargs):
                        return original_autocast(**{**kwargs, "cache_enabled": weight_cache})

                    with patch("torch.autocast", context):
                        output = scst_rollouts(
                            model,
                            pixels,
                            **options,
                            baseline=baseline,
                            sample_model_mode=mode,
                            use_kv_cache=kv_cache,
                            amp=True,
                        )
                    output.sampled.action_log_probs.sum().backward()
                    gradients = {
                        n: p.grad.detach().cpu().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None
                    }
                    missing = [
                        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
                    ]
                    nonfinite = [n for n, g in gradients.items() if not torch.isfinite(g).all()]
                    case = {
                        "baseline": baseline,
                        "sample_mode": mode,
                        "kv_cache": kv_cache,
                        "weight_cache": weight_cache,
                        "missing_gradients": missing,
                        "nonfinite_gradients": nonfinite,
                    }
                    assert not missing and not nonfinite, case
                    if reference is None:
                        reference = (output.sampled.token_ids.clone(), gradients)
                    else:
                        assert torch.equal(reference[0], output.sampled.token_ids)
                        case["matches_uncached_tokens"] = True
                        case["max_gradient_absolute_difference"] = max(
                            (gradients[n] - reference[1][n]).abs().max().item() for n in gradients
                        )
                        for n in gradients:
                            torch.testing.assert_close(
                                gradients[n], reference[1][n], rtol=0.03, atol=0.002
                            )
                        case["gradients_close_to_uncached_reference"] = True
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
                    del output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Audit saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
