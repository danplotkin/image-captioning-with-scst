"""Evaluate validation-selected XE/REINFORCE/SCST on one fixed test corpus.

Run from the repository root. Existing outputs require --resume and must match
the checkpoint, source, data, precision, and decoding protocol exactly.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

from tqdm import tqdm

from scst_captioner import cli
from scst_captioner.checkpointing import sha256_file
from scst_captioner.config import load_config
from scst_captioner.reproducibility import collect_environment

RUNS = {
    "xe": "flickr30k-xe-finetune-runpod",
    "reinforce": "flickr30k-reinforce-runpod-continued",
    "scst": "flickr30k-scst-runpod-continued",
}


def source_identity():
    files = [Path("pyproject.toml"), Path("uv.lock")]
    for folder in ("src", "scripts", "configs"):
        files.extend(
            p for p in Path(folder).rglob("*") if p.suffix in (".py", ".toml", ".txt", ".sh")
        )
    return {str(p): sha256_file(p) for p in sorted(set(files))}


def validate_cell(payload, predictions, expected_ids, checkpoint_hash, decoding, amp):
    provenance = payload["provenance"]
    context = provenance["context"]
    if [row["image_id"] for row in predictions] != expected_ids:
        raise ValueError("Prediction image IDs do not match the ordered test split")
    if context["checkpoint_sha256"] != checkpoint_hash or context["decoding"] != decoding:
        raise ValueError("Evaluation cell has a different checkpoint or decoder")
    precision = "cuda_amp_fp16" if amp else "fp32"
    if context["inference_precision"] != precision or not context["use_kv_cache"]:
        raise ValueError("Evaluation cell has a different precision/cache protocol")
    if provenance["num_images"] != len(expected_ids) or provenance["reference_counts"] != [5]:
        raise ValueError("Evaluation cell has incomplete image/reference coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final-evaluation/test"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use FP16 only after cache validation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.amp and args.device != "cuda":
        parser.error("--amp requires CUDA")
    configs = {}
    for method in RUNS:
        name = "reinforce" if method == "reinforce" else "scst"
        config = load_config(f"configs/flickr30k-{name}-continue.toml")
        configs[method] = dataclasses.replace(
            config,
            runtime=dataclasses.replace(
                config.runtime, device=args.device, amp=args.amp, use_kv_cache=True
            ),
        )
    paths = {method: args.artifacts_root / run / "best.pt" for method, run in RUNS.items()}
    hashes = {method: sha256_file(path) for method, path in paths.items()}
    sources = source_identity()
    identity = {
        "checkpoints": hashes,
        "sources": sources,
        "configs": {k: v.to_dict() for k, v in configs.items()},
        "split_sha256": sha256_file(configs["xe"].data.split_manifest),
    }
    root = args.output_dir
    identity_path = root / "identity.json"
    if root.exists():
        if not args.resume or not identity_path.exists():
            raise ValueError(
                "Output exists; use a new directory or --resume with matching identity"
            )
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Resume source/checkpoint/config/data identity differs")
    else:
        root.mkdir(parents=True)
        cli.write_json(identity_path, identity)
    setup = cli._evaluation_setup(configs["xe"])
    device, manifest, tokenizer, train_records, test_records, loader, model = setup
    if len(test_records) != 1000:
        raise ValueError("Final Flickr30k evaluation requires the 1,000-image test split")
    expected_ids = [r.image_id for r in test_records]
    summaries = {}
    for method in RUNS:
        summaries[method] = cli._preflight_checkpoint(
            str(paths[method]),
            expected_stage="xe" if method == "xe" else "scst",
            config=configs[method],
            manifest=manifest,
        )
        if method != "xe":
            parent = cli._recorded_xe_parent_sha256(summaries[method]["provenance"])
            if parent != hashes["xe"]:
                raise ValueError(f"{method} does not descend from the supplied XE checkpoint")
    environment = collect_environment(Path.cwd())
    cells = {}
    for method in RUNS:
        cells[method] = {}
        for decoding in ("greedy", "beam"):
            destination = root / method / decoding
            metrics_file = destination / "metrics.json"
            prediction_file = destination / "predictions.json"
            if metrics_file.exists():
                payload = json.loads(metrics_file.read_text())
                predictions = json.loads(prediction_file.read_text())
                if sha256_file(prediction_file) != payload["prediction_artifact"]["sha256"]:
                    raise ValueError(f"Prediction hash mismatch: {destination}")
            else:
                print(f"Evaluating {method} / {decoding}: 1000 test images", flush=True)
                start = time.perf_counter()
                result, predictions, summary, objective = cli._evaluate_checkpoint(
                    checkpoint_path=str(paths[method]),
                    decoding=decoding,
                    config=configs[method],
                    device=device,
                    manifest=manifest,
                    tokenizer=tokenizer,
                    train_records=train_records,
                    test_records=test_records,
                    loader=tqdm(loader, desc=f"{method}/{decoding}", mininterval=10),
                    model=model,
                    expected_stage="xe" if method == "xe" else "scst",
                )
                destination.mkdir(parents=True, exist_ok=True)
                predictions = cli._prediction_payload(predictions, objective)
                cli.write_json(prediction_file, predictions)
                payload = result.to_dict()
                payload.update(
                    training_objective=objective,
                    checkpoint_metadata=summary,
                    resolved_config=configs[method].to_dict(),
                    environment=environment,
                    elapsed_seconds=time.perf_counter() - start,
                    prediction_artifact={
                        "path": prediction_file.name,
                        "sha256": sha256_file(prediction_file),
                    },
                )
                cli.write_json(metrics_file, payload)
            validate_cell(payload, predictions, expected_ids, hashes[method], decoding, args.amp)
            cells[method][decoding] = payload
            print(
                f"Completed {method}/{decoding}: {payload['metrics']}; "
                f"training CIDEr-D={payload['training_objective']['mean']}",
                flush=True,
            )
    protocol_hashes = {
        p["provenance"]["comparison_protocol_sha256"]
        for row in cells.values()
        for p in row.values()
    }
    if len(protocol_hashes) != 1:
        raise ValueError("Evaluation cells do not share the same corpus/metric protocol")
    cli.write_json(
        root / "comparison.json",
        {
            "identity": identity,
            "environment": environment,
            "metric_corpus_protocol_sha256": next(iter(protocol_hashes)),
            "cells": cells,
        },
    )
    rows = [
        {
            "model": method,
            "decoding": decoder,
            **p["metrics"],
            "training_cider_d": p["training_objective"]["mean"],
        }
        for method, values in cells.items()
        for decoder, p in values.items()
    ]
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
