# Image Captioning with Self-Critical Sequence Training

A reproducible, CPTR-inspired image captioner built with a pretrained ViT encoder,
an autoregressive Transformer decoder, cross-entropy (XE) pretraining, and
self-critical sequence training (SCST).

This repository is now a normal Python package and CLI. The original Colab notebook
is preserved unchanged under [`notebooks/legacy/`](notebooks/legacy/ImageCaptioner.ipynb)
for historical inspection; it is not the executable source of truth.

## Project status

The implementation migration and offline correctness tests are complete. A fresh
Flickr8K training/evaluation run is still required before publishing new model
scores because the dataset and historical `cptr.pt`/`cptr_scst.pt` files are not in
this repository.

The old README/notebook scores are **historical and unverified**. The notebook was
executed out of order, separate model objects shared one global ViT and decoder,
the latest recorded SCST run was initialized after an older SCST checkpoint had
already mutated that shared decoder, and its evaluations mixed single-reference
greedy and multi-reference beam protocols. Those numbers are not regression
targets for the corrected code.

The detailed evidence and acceptance gates are in:

- [`docs/AUDIT.md`](docs/AUDIT.md)
- [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md)
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)
- [`docs/RESEARCH_ALIGNMENT.md`](docs/RESEARCH_ALIGNMENT.md)
- [`docs/CHECKPOINT_COMPATIBILITY.md`](docs/CHECKPOINT_COMPATIBILITY.md)

## What was corrected

- The greedy self-critical baseline always uses `model.eval()` and no gradient
  recording. The sampled branch defaults to eval mode while retaining autograd;
  train-mode sampling is an explicit configuration option.
- SCST batches one image once and scores both rollouts against all references.
- The first EOS action is included in the policy loss and every later action is
  excluded with an explicit boolean mask.
- Batched generation preserves `[batch, time]` for batch size one.
- PAD/BOS/MASK/UNK and other non-EOS special tokens are excluded from decoding.
- Every model instance owns an independent encoder and decoder while retaining
  the notebook's state-dict key layout for legacy weight loading.
- Flickr8K splits are image-level, ordered, persisted, disjoint, validated, and
  tied to caption/source hashes. Short references are no longer silently deleted.
- Augmentation happens on PIL images before the pinned ViT image processor
  rescales and normalizes them.
- Research evaluation uses one fixed test manifest and the COCO caption suite,
  alongside the exact configured training reward.
- Checkpoints contain stage, resolved config, optimizer/scheduler/scaler state,
  RNG and DataLoader state, selection metric, data identity, and code/environment
  provenance. Raw notebook weights remain loadable but cannot exactly resume.

## Research scope

This is a **CPTR-inspired adaptation**, not an exact CPTR reproduction. CPTR trains
its pretrained ViT-initialized encoder end to end on MS COCO; this project's
default freezes a pinned pretrained ViT and targets Flickr8K. SCST uses cached,
multi-reference CIDEr-D by default. NLTK METEOR remains an optional reward.

The original SCST paper defines the baseline as the reward from the model's own
test-time inference algorithm. Greedy decoding is canonical and remains the
training baseline; beam-3 is a separate evaluation factor. See the
[SCST paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf),
[CPTR paper](https://arxiv.org/pdf/2101.10804), and canonical
[COCO caption evaluator](https://github.com/tylin/coco-caption).

## Installation

Python 3.11–3.13 is supported. With [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra eval
uv run pytest
uv run ruff check .
```

The `eval` extra supplies the default CIDEr-D reward and COCO metrics.
COCO METEOR/PTB tokenization requires a working Java runtime; the CIDEr-D training
reward does not. Only the optional `nltk_meteor` reward requires explicitly
installed WordNet/OMW data and never downloads it implicitly:

```bash
uv run python -m nltk.downloader wordnet omw-1.4
```

The committed `uv.lock` fixes the Python dependency graph. The Hugging Face ViT,
tokenizer, and image-processor revisions are pinned in
[`configs/default.toml`](configs/default.toml).

## Data layout and split preparation

Place the external Flickr8K files at the configured paths (they are intentionally
gitignored):

```text
data/flickr8k/
  Images/
  captions.txt
  Flickr_8k.trainImages.txt   # optional but preferred
  Flickr_8k.devImages.txt     # optional but preferred
  Flickr_8k.testImages.txt    # optional but preferred
```

Create the conventional split manifest when the three distributed split files are
available:

```bash
uv run scst-captioner prepare-data \
  --official-train data/flickr8k/Flickr_8k.trainImages.txt \
  --official-validation data/flickr8k/Flickr_8k.devImages.txt \
  --official-test data/flickr8k/Flickr_8k.testImages.txt

uv run scst-captioner validate-data
```

Without those files, `prepare-data` creates and records a deterministic 80/10/10
image-level split from sorted image IDs and seed 42. The manifest records the
algorithm version, exact fractions, filtering policy, source hashes, excluded
images, and ordered IDs.

## Training

```bash
# Teacher-forced pretraining
uv run scst-captioner train-xe \
  --config configs/default.toml \
  --output-dir artifacts/xe

# Corrected image-level, all-reference SCST
uv run scst-captioner train-scst \
  --config configs/default.toml \
  --xe-checkpoint artifacts/xe/best.pt \
  --output-dir artifacts/scst
```

Both commands save `best.pt`, `last.pt`, `history.json`, `resolved_config.json`,
and `provenance.json`. Resume from `last.pt` with `--resume`; SCST does not require
`--xe-checkpoint` when resuming.

The default reward is `cider_d`: cached, symmetrically normalized CIDEr-D with
completion-sensitive EOS handling. Both training and validation checkpoint
selection use this reward. Set `scst.reward = "nltk_meteor"` for a METEOR run.
Start a new CIDEr-D run from an XE checkpoint instead of resuming a METEOR SCST
checkpoint, whose reward protocol differs. Evaluation still reports BLEU-1–4,
METEOR, ROUGE-L, and CIDEr, plus the exact configured training reward. Set
`scst.sample_model_mode = "train"` only for an explicitly labeled dropout-policy
ablation.

The CIDEr-D adapter fixes both document frequencies and the IDF corpus size to
the training split, so rewards are independent of batch size. Its v2 reward
protocol rejects older CIDEr-D checkpoints that used a batch-dependent IDF
denominator; start a fresh SCST run from XE for the corrected scores.

## Evaluation

The required apples-to-apples comparison is generated automatically:

```bash
uv run scst-captioner evaluate-matrix \
  --config configs/default.toml \
  --xe-checkpoint artifacts/xe/best.pt \
  --scst-checkpoint artifacts/scst/best.pt \
  --output-dir artifacts/evaluation-matrix
```

It produces all four cells:

| checkpoint | greedy | beam-3 |
|---|---:|---:|
| XE-selected | COCO suite + exact configured reward | COCO suite + exact configured reward |
| SCST-selected | COCO suite + exact configured reward | COCO suite + exact configured reward |

The command rejects wrong stages, mismatched model/reward/data protocols, split
hash mismatches, raw/unverified legacy weights, and an SCST checkpoint whose
recorded parent is not the supplied XE checkpoint. It stores predictions with
completion state, per-image scores, checkpoint hashes, full protocol hashes,
resolved config, and environment metadata.

Caption one image with either decoder:

```bash
uv run scst-captioner caption path/to/image.jpg \
  --checkpoint artifacts/scst/best.pt \
  --decoding beam
```

## Legacy checkpoints

The model preserves the notebook parameter paths, including
`backbone.backbone.*`, `positional_embedding.*`, `decoder.layers.*`, and `fc.*`.
A raw historical state dict can therefore be strictly wrapped:

```bash
uv run scst-captioner convert-checkpoint \
  --input cptr.pt \
  --output artifacts/legacy-cptr.pt \
  --stage xe
```

Conversion records unknown lineage fields; it does not invent a split, optimizer,
RNG state, or scientific validity. Unverified legacy artifacts are blocked from
the four-way research matrix and require an explicit override for one-off
historical evaluation.

## Validation boundary

The test suite is hermetic: it injects tiny encoders, tokenizers, scorers, images,
and datasets, so it does not download a ViT or Flickr8K. It covers data leakage,
manifest stability, model independence, batch-one decoding, first-EOS behavior,
dropout modes, REINFORCE gradients, cached multi-reference rewards, checkpoint
round trips, strict metric alignment, and tiny CPU XE/SCST optimizer steps.

Fresh real-data training, the real four-cell metric matrix, repeated seeds, and
confidence intervals are Gate 6 in the migration plan. Until those artifacts
exist, this repository makes no replacement performance claim.

## Dataset and licensing note

Flickr images are external assets with their own owners/licensing terms and are
not relicensed by this repository. A code license has not yet been selected; add
one deliberately before redistributing the project as open source.
