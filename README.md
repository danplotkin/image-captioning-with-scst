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
  Identical captions from different annotators retain their reference slots.
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

From the project root, run the repeatable setup script:

```bash
uv sync --extra eval
uv run --with kagglehub python scripts/setup_flickr8k.py
```

The script downloads [Flickr8K version 1](https://www.kaggle.com/datasets/adityajn105/flickr8k)
(approximately 1.1 GB) and the conventional split lists from the
[Flickr8K annotation archive mirror](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k).
It verifies pinned SHA256 hashes for the captions and split lists, validates the
images and all five references per image, and creates `data/splits/flickr8k.json`:

| Split | Images |
| --- | ---: |
| Training | 6,000 |
| Validation | 1,000 |
| Test | 1,000 |
| Excluded from conventional lists | 91 |

All captions for an image stay in the same split. Reuse this manifest for XE,
SCST, and evaluation. Dataset files and generated manifests are gitignored;
archive the manifest with experiment artifacts when sharing results.

Rerunning the script reuses existing assets and verifies the saved manifest
without rewriting it. Once all data and lists are present, the Python script
requires no network access. Changed captions, split lists, or a conflicting
manifest cause an error rather than silently changing an experiment's data.
For an existing custom split, preserve its manifest with the associated experiment
artifacts before removing it and rerunning setup.

```text
data/flickr8k/
  Images/
  captions.txt
  Flickr_8k.trainImages.txt
  Flickr_8k.devImages.txt
  Flickr_8k.testImages.txt
data/splits/
  flickr8k.json
```

To revalidate independently:

```bash
uv run scst-captioner validate-data
```

For a deliberate custom split instead of the conventional benchmark, the CLI
also supports `uv run scst-captioner prepare-data`. Without official split-file
arguments it creates a deterministic 80/10/10 image-level split with seed 42.
The conventional setup script rejects such a manifest instead of overwriting it.

### Flickr30k

The same model, XE trainer, SCST trainer, and evaluator support Flickr30k.
Use the dedicated setup script and configuration:

```bash
uv run --with kagglehub python scripts/setup_flickr30k.py
uv run scst-captioner validate-data --config configs/flickr30k.toml
uv run scst-captioner train-xe --config configs/flickr30k.toml
uv run scst-captioner train-scst --config configs/flickr30k.toml \
  --xe-checkpoint artifacts/flickr30k/xe/best.pt
```

Setup downloads the pinned [Flickr30k image mirror, version 1](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
through KaggleHub and the [Karpathy annotation archive](https://cs.stanford.edu/people/karpathy/deepimagesent/).
Allow several GB for the images plus download/extraction space. If Kaggle requires
authentication in your environment, authenticate with KaggleHub and rerun.
Captions and split assignments come from the same SHA256-verified annotation
JSON; the mirror's caption file is not used. Setup converts the five raw
references per image to the CSV format consumed by the existing loader, retaining
punctuation, case, and duplicate references.

The original Karpathy annotation release contains **29,000 training, 1,014
validation, and 1,000 test images** (31,014 total). Additional images in the
image mirror are unused. All references for each image stay together. The manifest
records `dataset = "flickr30k"`, annotation/caption hashes, and an image-content
fingerprint. Reruns verify existing files and manifests without overwriting
conflicting data. Once the required assets are present, setup works offline.

Data lives under `data/flickr30k/`. KaggleHub downloads into the dedicated
`data/flickr30k/kaggle/` directory, keeping its download destination separate
from our annotations and generated captions. The manifest lives at
`data/splits/flickr30k.json`. The config writes training outputs under
`artifacts/flickr30k/xe/` and `artifacts/flickr30k/scst/`. The model settings and
batch sizes match the default configuration. Training has about five times as
many examples per epoch as Flickr8K; batch memory requirements remain similar.
Start a fresh XE run for this dataset, then initialize SCST from its best XE
checkpoint. Existing Flickr8K checkpoints cannot exactly resume a Flickr30k run
because checkpoints validate dataset identity.

The playground also uses Flickr30k when `CONFIG_PATH` is set to
`"configs/flickr30k.toml"`. Use the setup script to recreate the benchmark split;
`prepare-data` without split files creates a custom random partition.

## Visual training playground

Open [`playground.ipynb`](notebooks/playground.ipynb) to inspect real Flickr8K images through
the configured training pipeline:

```bash
uv sync --extra eval --extra notebook
uv run jupyter lab notebooks/playground.ipynb
```

Change `CONFIG_PATH` and the preview controls in the first code cell, then run
all cells. The notebook shows augmentation stages, normalized image channels,
ViT patches, caption tokens and shifted targets, actual configured XE/SCST
batches, attention/loss masks, and model output shapes. It uses the production
data loaders and model code. The model forward preview runs on a small CPU batch
without optimizer updates; the configured full-batch shapes are shown separately.
An optional SCST rollout shows sampled log probabilities and EOS masks. The
caption decoder is newly initialized, so preview captions are not quality results.
Pinned Hugging Face components may download on first use.

## CPU smoke run

After setting up Flickr8K, prepare the small real-image fixture and run XE:

```bash
uv run python scripts/setup_smoke.py
uv run scst-captioner validate-data --config configs/smoke.toml
uv run scst-captioner train-xe --config configs/smoke.toml
```

The repeatable fixture takes the first 16 training, 4 validation, and 4 test
images from their existing benchmark splits, retaining all five captions per
image. It writes separate files under `data/smoke/`. The config uses CPU,
batch size 2, a frozen pretrained ViT, a one-layer 64-wide decoder, and one XE
epoch (40 training batches and 10 validation batches). Outputs go to
`artifacts/smoke/xe/`. The 12-token sequence budget is deliberately short.
This checks the pipeline; use the default configuration for the full experiment.
For smoke SCST, keep `--config configs/smoke.toml` and initialize from
`artifacts/smoke/xe/best.pt`.

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

Both commands automatically export `loss_curve.png` in the run directory after
each completed epoch, so it is ready when training finishes or XE stops early.
XE plots training and validation cross-entropy; SCST plots training REINFORCE
loss (its validation metric is reward, not loss). Resumed runs include the earlier
epochs. The plots also work on headless servers without opening a window.

Before a fresh SCST run, the loaded XE policy is evaluated once on the validation
set using greedy decoding and the configured training reward. This fixed score
is saved in `xe_validation.json` and in each checkpoint. Epoch summaries and
`history.json` report the current validation reward and its change from XE;
summaries also report the best change so far. `reward_curve.png` plots validation
reward against a horizontal XE reference line. Resuming preserves the original
XE score without another baseline evaluation. Older checkpoints without that
score still resume, but the XE comparison is marked unavailable. Best SCST
checkpoint selection continues to use validation reward, even if no epoch beats XE.

In an interactive terminal, Rich progress bars show completed batches, elapsed
time, ETA, and running metrics for training and validation. XE displays token-weighted
loss and accuracy; SCST displays image-weighted loss and sampled/baseline rewards.
Epoch summaries remain available when output is redirected, with animated bars disabled.

CUDA AMP skips updates with overflowing gradients and adjusts its loss scale.
Update counters and the XE scheduler advance only when an optimizer update
occurs; the SCST epoch scheduler advances when the epoch contains an update.
Nonfinite full-precision gradients and overflow of the norm of otherwise finite
gradients remain errors.

The default reward is `cider_d`: cached, symmetrically normalized CIDEr-D with
completion-sensitive EOS handling. Both training and validation checkpoint
selection use this reward. Set `scst.reward = "nltk_meteor"` for a METEOR run.
Start a new CIDEr-D run from an XE checkpoint instead of resuming a METEOR SCST
checkpoint, whose reward protocol differs. Evaluation still reports BLEU-1–4,
METEOR, ROUGE-L, and CIDEr, plus the exact configured training reward. Set
`scst.sample_model_mode = "train"` only for an explicitly labeled dropout-policy
ablation.

The CIDEr-D adapter fixes both document frequencies and the IDF corpus size to
the training split, so rewards are independent of batch size. Caption decoding
uses explicit cleanup for conventional apostrophe suffixes and single interword
hyphens. Reward text uses the same punctuation boundaries for predictions,
references, and the IDF corpus, so WordPiece spacing cannot penalize an exact
target-token sequence. This is recorded as CIDEr-D tokenization v3; optional
NLTK METEOR string normalization is v2. Standard COCO reference text is preserved.

Resume and evaluation reject SCST checkpoints with older reward/decoding
protocols, including the earlier batch-dependent IDF denominator. Start a fresh
SCST run from an XE checkpoint for the corrected reward protocol.

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

## Autoregressive KV cache

Greedy decoding and eval-mode SCST sampling use a per-rollout KV cache by default.
Each decoder layer reuses caption attention keys/values and projected image
keys/values. The sampled cache retains autograd history for the policy gradient.
The default 768-wide encoder/decoder retains its existing checkpoint layout.

Full-sequence training and cached decoding share the same `CaptionDecoderLayer`
implementation in `src/scst_captioner/decoder.py`, built from PyTorch attention,
normalization, and linear modules.

Image memory is the raw encoder output. Each cross-attention layer sets its
`kdim` and `vdim` to the encoder width and projects image keys/values directly
into the decoder width, without a separate memory-projection layer.

Set `use_kv_cache = false` under `[runtime]` in your TOML config to compare with
full-prefix decoding. Direct `greedy_decode`, `sample_decode`, and `scst_rollouts`
calls accept the same keyword. Train-mode sampling automatically uses full
prefixes to preserve its dropout behavior; beam search currently remains uncached.
Small floating-point differences can occasionally change token choices.

To verify a saved XE model using local pretrained assets and validation images:

```bash
uv run python scripts/verify_kv_cache.py --checkpoint artifacts/xe/best.pt
```

This strictly loads the saved weights, checks greedy token/mask equality, compares
seeded samples, and checks fixed-sample SCST logits, loss, image-memory gradients,
and every participating parameter gradient. It also times CPU greedy decoding
(excluding image encoding) and writes `artifacts/kv_cache_verification.json`.
The default check uses 16 validation images; it is not a full validation metric
run or a CUDA memory/performance benchmark.

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

### SCST versus vanilla REINFORCE

Both methods use the same `PolicyGradientTrainer` and `reinforce_loss`. Configure
`scst.baseline = "greedy"` for SCST (sample reward minus greedy reward), or
`scst.baseline = "none"` for vanilla REINFORCE (raw sample reward, zero baseline).
Vanilla REINFORCE skips greedy generation and scoring during training. Both
methods still evaluate greedy captions on validation and select the best
checkpoint by validation reward. Their training losses have different meanings;
compare validation rewards and caption metrics rather than raw loss magnitudes.

The paired Flickr30k configs retain the fine-tuned model's last four trainable ViT
blocks and final LayerNorm. They match the fine-tuned XE parent's saved model,
data, and XE protocol, and differ only in the RL baseline. Start each run from
the **same** fine-tuned XE checkpoint, never from the other RL run:

```bash
uv run scst-captioner train-rl \
  --config configs/flickr30k-scst.toml \
  --xe-checkpoint artifacts/flickr30k-xe-finetune-runpod/best.pt

# After the first command succeeds, independently initialize vanilla REINFORCE.
uv run scst-captioner train-rl \
  --config configs/flickr30k-reinforce.toml \
  --xe-checkpoint artifacts/flickr30k-xe-finetune-runpod/best.pt
```

Outputs default to `artifacts/flickr30k-policy-gradient/scst/` and
`artifacts/flickr30k-policy-gradient/reinforce/`. Each process seeds itself from
the config. Keep equal update budgets for the initial baseline ablation; measure
wall-clock time separately because vanilla REINFORCE omits a rollout. Multiple
seeds are needed to assess variation beyond this first paired run.

`train-scst` remains an alias-compatible entrypoint, and the `[scst]` config
section and checkpoint `stage = "scst"` remain for compatibility with existing
checkpoints and evaluation tools. The baseline in resolved config and
`policy_gradient_method` in provenance identify the actual method. Resume rejects
switching the baseline. `SCSTTrainer` remains a compatibility alias for
`PolicyGradientTrainer`; existing SCST callers keep the greedy default.

### XE fine-tuning with partial ViT unfreezing

Keep `configs/flickr30k.toml` for the original frozen-encoder experiment. To
initialize a new XE run from its best checkpoint:

```bash
uv run scst-captioner train-xe --config configs/flickr30k-xe-finetune.toml
```

The new, standalone config trains the final four ViT blocks and final LayerNorm
at `xe.encoder_learning_rate = 0.000001`, while the caption decoder trains at
`xe.learning_rate = 0.00001`. Patch embeddings and earlier blocks stay frozen
and in eval mode. It runs up to five epochs with patience two, writing under
`artifacts/flickr30k-xe-finetune/xe`. Ensure `xe.init_checkpoint` points to the
original run's `best.pt`; training does not download that checkpoint.

`xe.init_checkpoint` loads model weights, validates the XE stage, architecture,
dataset and preprocessing, and records the parent path/hash. Optimizer,
scheduler, scaler, history and early stopping start fresh. To resume this new
run, make a copy of its config without `xe.init_checkpoint` and use `--resume`
with its `last.pt`. All training settings must still match. Initialization and
resume cannot be requested together.

New fields are optional. Without `model.encoder_trainable_blocks`, the existing
`freeze_encoder` behavior is preserved (true freezes everything; false trains
everything). With a block count, `freeze_encoder` must be false to train any
blocks; zero freezes all blocks. `encoder_train_final_layer_norm` defaults to
false and controls the final norm in partial mode. A count cannot exceed the
backbone's number of blocks. Omitting `xe.encoder_learning_rate` preserves the
original single optimizer group. Older checkpoints receive these new defaults
during compatibility checks without modifying their saved metadata.

Use the fine-tuned XE checkpoint as the next SCST parent. SCST retains its own
`scst.learning_rate`; the separate encoder rate described here applies to XE.
