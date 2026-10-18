"""Build reproducible charts and caption panels from saved runs and test predictions."""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import textwrap
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from scst_captioner import cli
from scst_captioner.config import load_config

METHODS = ("xe", "reinforce", "scst")
COLORS = {"xe": "#64748b", "reinforce": "#2563eb", "scst": "#db6725"}
LABELS = {"xe": "XE", "reinforce": "REINFORCE", "scst": "SCST"}


def read_json(path):
    return json.loads(Path(path).read_text())


def continued_history(original, continued):
    if continued[: len(original)] != original:
        raise ValueError("Continuation history does not preserve the original prefix")
    if [row["epoch"] for row in continued] != list(range(1, len(continued) + 1)):
        raise ValueError("History has missing, duplicate, or out-of-order epochs")
    return continued


def select_examples(test_ids, count=12, seed=42):
    return random.Random(seed).sample(sorted(test_ids), count)


def figure(size):
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def validation_chart(artifacts, output):
    fig = figure((9, 5))
    ax = fig.subplots()
    curves = {}
    baseline = None
    for method in ("reinforce", "scst"):
        run = artifacts / "flickr30k-policy-gradient" / method
        history = continued_history([], read_json(run / "history.json"))
        xe = read_json(run / "xe_validation.json")["value"]
        if baseline is not None and xe != baseline:
            raise ValueError("RL runs use different XE validation baselines")
        baseline = xe
        best = max(history, key=lambda row: row["validation_reward"])
        curves[method] = {
            "history": history,
            "best_epoch": best["epoch"],
            "best_validation_reward": best["validation_reward"],
        }
        ax.plot(
            [r["epoch"] for r in history],
            [r["validation_reward"] for r in history],
            color=COLORS[method],
            marker="o",
            markersize=3,
            label=LABELS[method],
        )
        ax.scatter(
            [best["epoch"]],
            [best["validation_reward"]],
            color=COLORS[method],
            marker="*",
            s=160,
            zorder=5,
        )
    ax.axhline(baseline, color=COLORS["xe"], linestyle="--", label="XE baseline")
    ax.set(
        title="Flickr30k validation: saved experiment histories",
        xlabel="RL epoch",
        ylabel="Greedy CIDEr-D (fixed training-corpus IDF)",
        xticks=sorted({1, *range(5, max(len(c["history"]) for c in curves.values()) + 1, 5)}),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.set_layout_engine("constrained", rect=(0, 0.06, 1, 0.94))
    fig.text(
        0.02,
        0.015,
        "Checkpoints are selected using validation, independently of test scores.",
        fontsize=10,
    )
    fig.savefig(output / "validation-reward.png", dpi=180)
    fig.clear()
    payload = {"xe_validation_reward": baseline, "methods": curves}
    (output / "validation-results.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def test_chart(comparison, output):
    metrics = [
        ("CIDEr", "COCO CIDEr"),
        ("training_cider_d", "Training-objective CIDEr-D"),
        ("Bleu_1", "BLEU-1"),
        ("Bleu_4", "BLEU-4"),
        ("METEOR", "METEOR"),
        ("ROUGE_L", "ROUGE-L"),
    ]
    fig = figure((12, 7))
    axes = fig.subplots(2, 3).flat
    summary = []
    for method in METHODS:
        for decoding in ("greedy", "beam"):
            cell = comparison["cells"][method][decoding]
            summary.append(
                {
                    "model": method,
                    "decoding": decoding,
                    **cell["metrics"],
                    "training_cider_d": cell["training_objective"]["mean"],
                }
            )
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for index, method in enumerate(METHODS):
            values = [row[metric] for row in summary if row["model"] == method]
            ax.bar(
                [index - 0.18, index + 0.18], values, width=0.32, color=COLORS[method], alpha=0.9
            )
            ax.patches[-1].set_hatch("///")
        ax.set(
            title=title, xticks=[0, 1, 2], xticklabels=[LABELS[m] for m in METHODS], ylim=(0, None)
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Flickr30k test • 1,000 images • solid: greedy / hatched: beam-3")
    fig.set_layout_engine("constrained", rect=(0, 0.05, 1, 0.95))
    fig.text(
        0.02,
        0.01,
        "Compare methods within the same decoding protocol.",
        fontsize=10,
    )
    fig.savefig(output / "test-metrics.png", dpi=180)
    fig.clear()
    (output / "test-results.json").write_text(json.dumps(summary, indent=2) + "\n")


def gallery(root, config, test_ids, output, docs_output):
    records, _ = cli._load_corpus(config)
    by_id = {r.image_id: r for r in records}
    selected = select_examples(test_ids)
    predictions = {
        m: {
            d: {r["image_id"]: r for r in read_json(root / m / d / "predictions.json")}
            for d in ("greedy", "beam")
        }
        for m in METHODS
    }
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, image_id in enumerate(selected):
        record = by_id[image_id]
        captions = {
            m: {d: predictions[m][d][image_id] for d in ("greedy", "beam")} for m in METHODS
        }
        example = {
            "image_id": image_id,
            "references": list(record.captions),
            "predictions": captions,
            "panel": f"example-{index + 1:02d}.png",
        }
        fig = figure((12, 8))
        layout = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
        image_ax = fig.add_subplot(layout[0, 0])
        with Image.open(record.image_path) as source:
            image_ax.imshow(source.convert("RGB"))
        image_ax.axis("off")
        image_ax.set_title(image_id, fontsize=10)
        references_ax = fig.add_subplot(layout[0, 1])
        references_ax.axis("off")
        references_ax.text(
            0,
            1,
            "Human references\n\n" + "\n\n".join(textwrap.fill(c, 65) for c in record.captions),
            va="top",
            fontsize=10,
        )
        for col, decoding in enumerate(("greedy", "beam")):
            ax = fig.add_subplot(layout[1, col])
            ax.axis("off")
            ax.set_title("Greedy" if decoding == "greedy" else "Beam-3 (length penalty 0.7)")
            for row, method in enumerate(METHODS):
                item = captions[method][decoding]
                suffix = "" if item["completed"] else " [length limit]"
                ax.text(
                    0,
                    0.97 - row * 0.32,
                    LABELS[method] + "\n" + textwrap.fill(item["caption"] + suffix, 63),
                    va="top",
                    fontsize=10,
                    color=COLORS[method],
                )
        fig.savefig(output / example["panel"], dpi=150)
        fig.clear()
        examples.append(example)
    manifest = {
        "seed": 42,
        "selection": "random.Random(42).sample(sorted(test_ids), 12)",
        "examples": examples,
    }
    (output / "examples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cards = "\n".join(
        f'<figure><img src="{e["panel"]}" alt="Caption comparison for '
        f'{html.escape(e["image_id"])}"><figcaption>{html.escape(e["image_id"])}'
        "</figcaption></figure>"
        for e in examples
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        "<title>Flickr30k caption comparison</title><style>body{font-family:system-ui;"
        "max-width:1200px;margin:40px auto;background:#f8fafc}img{width:100%}"
        "figure{margin:25px 0}</style><h1>XE / REINFORCE / SCST</h1>"
        "<p>12 test images selected with seed 42, independently of scores. "
        "Captions come directly from the saved final test predictions.</p>" + cards
    )
    # The first three preselected examples keep the README compact, without cherry-picking.
    for example in examples[:3]:
        shutil.copy2(output / example["panel"], docs_output / example["panel"])
    shutil.copy2(output / "examples.json", docs_output / "examples.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--evaluation-dir", type=Path, default=Path("artifacts/final-evaluation/test")
    )
    parser.add_argument("--docs-output", type=Path, default=Path("docs/results"))
    parser.add_argument("--histories-only", action="store_true")
    args = parser.parse_args()
    args.docs_output.mkdir(parents=True, exist_ok=True)
    validation_chart(args.artifacts_root, args.docs_output)
    if args.histories_only:
        return
    comparison = read_json(args.evaluation_dir / "comparison.json")
    test_chart(comparison, args.docs_output)
    config = load_config("configs/flickr30k-scst.toml")
    test_ids = read_json(config.data.split_manifest)["splits"]["test"]
    gallery(
        args.evaluation_dir,
        config,
        test_ids,
        args.evaluation_dir.parent / "gallery",
        args.docs_output,
    )
    shutil.copy2(args.evaluation_dir / "comparison.csv", args.docs_output / "test-results.csv")


if __name__ == "__main__":
    main()
