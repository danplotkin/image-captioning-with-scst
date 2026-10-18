"""Audit saved SCST training for detached autocast weights; never update parameters."""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, ViTConfig, ViTModel

from scst_captioner.checkpointing import load_checkpoint_file, sha256_file
from scst_captioner.config import ModelConfig
from scst_captioner.generation import scst_rollouts
from scst_captioner.model import CPTRCaptioner
from scst_captioner.tokenization import resolve_special_token_ids


def main():
    torch.set_num_threads(4)
    path = Path("artifacts/flickr30k-scst-runpod-continued/best.pt")
    saved = load_checkpoint_file(path)
    config = ModelConfig(**saved["config"]["model"])
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
    ).cuda()
    model.load_state_dict(saved["model_state"])
    special = resolve_special_token_ids(tokenizer)
    data = saved["config"]["data"]
    image_id = json.loads(Path(data["split_manifest"]).read_text())["splits"]["validation"][0]
    with Image.open(Path(data["images_dir"]) / image_id) as image:
        pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].cuda()
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "checkpoint_sha256": sha256_file(path),
        "validation_image": image_id,
        "trainable_parameter_tensors": sum(p.requires_grad for p in model.parameters()),
        "saved_optimizer_state_count": len(saved["optimizer_state"]["state"]),
        "cases": [],
    }
    for baseline in ("greedy", "none"):
        for weight_cache in (True, False):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(42)
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=weight_cache):
                output = scst_rollouts(
                    model,
                    pixels,
                    bos_token_id=special.bos,
                    eos_token_id=special.eos,
                    max_new_tokens=3,
                    forbidden_token_ids=special.forbidden,
                    baseline=baseline,
                )
            output.sampled.action_log_probs.sum().backward()
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            report["cases"].append(
                {
                    "baseline": baseline,
                    "autocast_weight_cache": weight_cache,
                    "missing_gradients": missing,
                }
            )
            print(
                baseline,
                "weight_cache",
                weight_cache,
                "missing gradients",
                len(missing),
                flush=True,
            )
    missing = report["cases"][0]["missing_gradients"]
    xe = load_checkpoint_file("artifacts/flickr30k-xe-finetune-runpod/best.pt")
    report["missing_parameters_unchanged_from_xe"] = {
        n: torch.equal(saved["model_state"][n], xe["model_state"][n]) for n in missing
    }
    del saved
    reinforce = load_checkpoint_file("artifacts/flickr30k-reinforce-runpod-continued/best.pt")
    report["reinforce_optimizer_state_count"] = len(reinforce["optimizer_state"]["state"])
    report["same_parameters_changed_in_reinforce"] = {
        n: not torch.equal(reinforce["model_state"][n], xe["model_state"][n]) for n in missing
    }
    output = Path("artifacts/final-evaluation/scst-training-autocast-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Audit saved:", output, flush=True)


if __name__ == "__main__":
    main()
