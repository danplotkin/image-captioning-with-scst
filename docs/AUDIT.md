# Implementation audit

## Scope and verdict

This audit covers the repository at commit `062780e` and, in particular, the
saved source and outputs in `notebooks/legacy/ImageCaptioner.ipynb`. It is a code and experiment
protocol audit, not a new benchmark run. The Flickr8K files and the historical
`cptr.pt` and `cptr_scst.pt` checkpoints are not in the repository, so the
original training run cannot be reproduced or independently scored from the
checked-in files alone.

The central policy-gradient expression is correct:

\[
A_i = r(y_i^{sample}) - r(y_i^{greedy}), \qquad
L_{SCST} = -\frac{1}{B}\sum_i A_i
             \sum_t \log p_\theta(y_{i,t}^{sample}\mid y_{i,<t}^{sample}, x_i).
\]

The notebook samples autoregressively, retains the sampled-token log
probabilities, includes the first EOS action, excludes actions after EOS, and
uses the correct loss sign. The notebook is nevertheless not a faithful or
reproducible SCST experiment because its self-critical baseline is generated
with training-mode dropout, its reward uses one reference at a time, and its
reported evaluations change both reference and decoding protocols.

The appropriate project description is therefore:

> A CPTR-inspired captioning model with a frozen pretrained ViT encoder and a
> correct REINFORCE loss, undergoing migration to a reproducible,
> multi-reference SCST training and evaluation protocol.

## Evidence from the current notebook

- The file is approximately 4.56 MB, although cell source accounts for only
  about 47 KB. Approximately 3.81 MB is serialized output.
- It has 104 cells: 56 code cells and 48 Markdown cells.
- Saved execution counts are nonlinear. For example, SCST training is execution
  48, while cells positioned later in the notebook contain executions 38--45.
  Several supervised training/evaluation cells have no execution count while
  retaining outputs.
- The saved data outputs report 7,467 retained images and 23,890/5,975/7,470
  caption rows for train/validation/test.
- The latest saved final metric output is METEOR `0.4681`, while the README
  reports `0.4659`.
- Notebook metadata records an A100, while the README says the run used an L4.

These observations do not prove that a particular saved score is wrong. They
do show that the notebook is not sufficient evidence for reconstructing which
code, model object, checkpoint, split order, and environment produced it.

## Findings

| ID | Severity | Finding | Required resolution |
|---|---|---|---|
| A-01 | Critical | `SCSTTrainer._train_epoch` calls `model.train()` and then obtains the greedy baseline under `torch.no_grad()`. `no_grad()` does not disable positional or decoder dropout. | Generate the greedy baseline using the test-time model in `eval()` mode. The corrected default also samples in eval mode so both branches use the same inference-time policy. Test the mode observed by both branches. |
| A-02 | Critical | SCST uses the caption-row dataset and `single_meteor_score`, so every image is generated repeatedly and compared with one reference at a time. | Add an image-level dataset view that returns all references. Score sampled and greedy captions against the same full reference set. |
| A-03 | Critical | The ViT and Transformer decoder are global module instances assigned to every new `CPTR`. Nominally independent models therefore share parameters, and deleting one model does not release those globals. | Construct and own both modules inside each model instance. Preserve legacy state-dict paths and test that parameter storage is independent. |
| A-04 | High | Sampling and greedy decoding use unrestricted `.squeeze()`. Batch size one changes the public shape; sampling may reduce a token to a scalar and fail. | Preserve `[batch, time]` and `[batch, vocabulary]` dimensions. Test batch sizes one and greater than one through every decoder. |
| A-05 | High | The before/after values use greedy, single-reference evaluation, while the final value uses beam-3, multi-reference evaluation. | Produce the XE/SCST by greedy/beam evaluation matrix on one fixed image list and one reference protocol. |
| A-06 | High | Evaluation averages smoothed `sentence_bleu` values. This is not corpus BLEU and should not be compared with standard captioning results. | Use one pinned corpus/COCO-compatible evaluation backend. Report BLEU-1--4, METEOR, ROUGE-L, CIDEr, and optionally SPICE. |
| A-07 | High | The split is generated in memory and is not persisted with dataset/preprocessing hashes. | Write an ordered image-ID manifest, hash it, and assert pairwise disjointness and complete coverage. |
| A-08 | High | Checkpoints are raw state dictionaries without model/data metadata, optimizer state, RNG state, or safe/map-location-aware loading. | Support legacy weight loading and a versioned checkpoint envelope. Document that a raw checkpoint cannot exactly resume training. |
| A-09 | High | The notebook is executed out of order, contains environment mutations and downloads, and is mostly output. | Move executable logic into importable modules and CLI commands. Retain at most a thin demo notebook. |
| A-10 | High | README and code disagree about SCST learning rate/decay, hardware, final score, and beam scoring. | Generate reports from resolved configs and result artifacts; do not manually merge metrics from different runs. |
| A-11 | Medium | Augmentations occur after ViT resize/rescale/normalization. Hue and fill operations then act in normalized model space. | Apply stochastic image augmentation to the PIL image before deterministic ViT preprocessing. |
| A-12 | Medium | A frozen ViT is still recursively placed in training mode. | Keep a frozen encoder in eval mode and run it without gradient recording. |
| A-13 | Medium | Generated distributions allow PAD, BOS, MASK, and other non-EOS special tokens, while reward decoding removes them with `skip_special_tokens=True`. | Suppress nonsemantic special-token actions consistently in sample, greedy, and beam decoding. |
| A-14 | Medium | Validation averages per-batch averages, so a short final batch is overweighted. | Accumulate summed token losses/correct counts and summed image rewards with their true denominators. |
| A-15 | Medium | Caption parsing uses globals and removes an entire image when any caption has fewer than five words. | Use robust CSV parsing and explicit, versioned filtering policy. Preserve raw and normalized references separately. |
| A-16 | Medium | Beam-search documentation calls a length-normalized score raw `log P(y|x)`, and the notebook code implements a different penalty expression. | Keep raw cumulative log probability distinct from the declared search-only length penalty. Test the scoring equation. |
| A-17 | Medium | The final test image order is made through `set`, which is process-dependent. | Iterate the ordered split manifest and persist predictions keyed by image ID. |
| A-18 | Medium | SCST best-policy tracking deep-copies the full model, including the ViT. | Save an on-CPU state snapshot or atomic best checkpoint. |
| A-19 | Medium | CUDA autocast/scaling and a global `DEVICE` are embedded in training and generation code. | Resolve a runtime device once, pass tensors/models explicitly, and enable AMP only on supported devices. |
| A-20 | Claim accuracy | A frozen Hugging Face ViT adaptation is not an exact reproduction of the end-to-end CPTR training recipe. | Use “CPTR-inspired” or “CPTR adaptation” in project claims and document the deviations. |

## Split clarification

The notebook does **not** currently split by caption row. It shuffles image
keys, takes 80% as a train/validation pool, then assigns 20% of that pool to
validation before expanding each split into caption rows. The effective image
fractions are therefore approximately 64% train, 16% validation, and 20% test.
That protects against direct image leakage for this code path.

The remaining problem is reproducibility: the exact image IDs, their order,
the source captions hash, and the filtering policy were not saved. The new
manifest must make those properties inspectable and assert them at load time.

## Status of historical scores

All notebook and README scores are **historical and unverified**. They may be
retained in a clearly labeled history section, but they must not be used as the
acceptance target for the corrected implementation. In particular:

1. nonlinear notebook execution prevents linking outputs to a unique source
   state;
2. global decoder/encoder sharing permits cross-instance parameter
   contamination;
3. the test split manifest and historical checkpoints are absent;
4. single-reference greedy and multi-reference beam results are different
   protocols; and
5. README values do not match the latest notebook outputs/configuration.

Corrected results start a new experiment lineage. The first trustworthy result
must be generated by the protocol in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
and carry checkpoint, config, split, reference, code, and environment identity.

## Validation boundary

This audit was validated by inspecting the latest notebook source, its saved
outputs and metadata, the repository tree, and the README. It did not rerun
Flickr8K training or load a historical checkpoint because those artifacts are
not present. The phased gates in [MIGRATION_PLAN.md](MIGRATION_PLAN.md) define
what must pass before implementation or experimental claims are considered
complete.
