# Corrected experiment protocol

## Purpose

This document defines the only protocol under which new project results should
be reported. It separates the effect of SCST fine-tuning from the effect of
changing the decoder or reference set and records the information needed to
repeat a run.

Historical notebook and README scores do not satisfy this protocol. They are
unverified for the reasons in [AUDIT.md](AUDIT.md) and must not be inserted into
the result matrix below.

## Experimental question

For a fixed CPTR-inspired architecture, Flickr8K partition, preprocessing
pipeline, tokenizer, and inference procedure:

> Does self-critical sequence training with a multi-reference sequence reward
> improve held-out caption quality relative to the selected cross-entropy
> checkpoint?

The primary causal comparison is XE versus SCST under **greedy,
multi-reference** evaluation. Beam search is a separate decoder factor.

## Dataset and references

Parse captions into one canonical record per image:

```text
image_id, image_path, raw_references[], normalized_references[]
```

Requirements:

- retain raw references for standardized evaluation;
- use a declared normalization function for model training/reward text;
- never split after expanding into caption rows;
- do not silently discard an image because one reference is short;
- record source captions plus ordered image-content hashes; and
- record the number of images and reference-count distribution before and
  after every explicit filter.

XE may use one dataset row per image/reference pair to preserve the supervised
training schedule. SCST, validation reward, and test evaluation must use one
row per image with all references.

## Split protocol

Use an ordered, versioned image-ID manifest. The manifest records:

- schema and split-algorithm versions;
- dataset identifier and source hash;
- normalization/filter policy and hash;
- seed;
- ordered train/validation/test/excluded image IDs and exact seeded fractions; and
- a manifest hash.

At load time assert:

\[
Train \cap Validation = Train \cap Test = Validation \cap Test = \varnothing
\]

and that the union is the complete filtered canonical corpus. Test order is the
manifest order; never reconstruct it from a set.

The notebook's effective random fractions are 64% train, 16% validation, and
20% test. A legacy manifest generator may reproduce that algorithm for
historical checkpoint inspection. A corrected experiment may instead use a
declared standard Flickr8K partition or a stable new partition, but results from
different manifests are not directly comparable.

## Reproducibility controls

Set and record one run seed for Python, NumPy, Torch, all CUDA devices,
DataLoader generators, and DataLoader workers. Record whether deterministic
Torch algorithms are enabled. Because hardware and library kernels may still
introduce numeric differences, save predictions and checkpoints rather than
claiming that a seed alone guarantees bit-for-bit training.

Each run stores:

- resolved config;
- code revision and dirty-worktree status;
- Python, Torch, torchvision, Transformers, NLTK, metric-backend and CUDA
  versions;
- device/GPU identity and AMP/determinism settings;
- encoder/tokenizer identifiers and pinned revisions when available;
- source, preprocessing, reference, and split hashes;
- checkpoint and prediction SHA256 values; and
- all optimizer/scheduler/early-stopping decisions.

## Model protocol

The default model is explicitly an adaptation:

- pretrained `google/vit-base-patch16-384` visual encoder;
- frozen encoder unless a separate ablation says otherwise;
- DistilBERT uncased tokenizer;
- autoregressive Transformer decoder;
- sinusoidal positional encoding; and
- maximum sequence capacity declared by config.

Every model owns its modules. A frozen encoder remains in eval mode and runs
without gradients. Non-EOS special token IDs are suppressed during all
generation methods.

Any change to encoder freezing, tokenizer/vocabulary, decoder depth/width,
dropout, image resolution, caption normalization, or special-token policy
defines a different model/protocol and must receive a separate result row.

## Cross-entropy stage

Teacher-forced input contains BOS and excludes the final target token. Targets
exclude BOS and include EOS. Pad-token labels are ignored.

Report token-level cross entropy and accuracy as global sums divided by the
number of non-pad target tokens—not an unweighted mean of batch metrics.
Select the XE checkpoint using validation loss only, with patience and minimum
delta fixed before the run. Save `best` and `last` checkpoints separately.

The XE checkpoint is the initialization for corrected SCST. A historical SCST
checkpoint cannot substitute for this stage because its baseline was generated
under the old stochastic protocol.

## SCST stage

For image \(x_i\), sample caption \(y_i^s\) and obtain the greedy test-time
caption \(\hat y_i\). Compute both rewards against the same full reference set
\(R_i\):

\[
A_i = r(y_i^s, R_i) - r(\hat y_i, R_i).
\]

With \(m_{i,t}\) equal to one through the first EOS action and zero thereafter,
minimize:

\[
L_{SCST} = -\frac{1}{B}\sum_{i=1}^{B}
  \operatorname{stopgrad}(A_i)
  \sum_t m_{i,t}\log p_\theta(y_{i,t}^s\mid y_{i,<t}^s,x_i).
\]

Required execution semantics:

1. put the model in eval mode;
2. encode the image once when encoder/mode semantics permit;
3. run greedy decoding under `no_grad()`;
4. run categorical sampling with autograd enabled but dropout still disabled;
5. score both decoded strings against all references with the identical reward
   implementation;
6. detach the advantage explicitly;
7. apply gradient clipping and finite-value checks; and
8. restore caller mode only after the rollout/step boundary as required by the
   training engine.

Using eval mode does not disable autograd. The sampled token log probabilities
remain differentiable. The baseline caption and reward do not.

The default baseline is greedy. A beam baseline is an expensive, separately
named experiment; it is not required merely because beam is also evaluated.

## Reward protocol

The default reward is multi-reference CIDEr-D (`scst.reward = "cider_d"`).
Its production adapter case-folds and Treebank-tokenizes both sides, filters
COCO punctuation, caches full-train document frequencies once, appends `<eos>`
only to completed predictions, and records that tokenization/version. Training
and validation checkpoint selection use the same reward implementation.
Both document frequencies and their log-corpus-size denominator stay fixed to
the training split, including singleton validation batches. The v2 adapter
preserves COCO's clipping, Gaussian length penalty, and 10x score scaling while
preventing its scorer from replacing the corpus size with the batch size.
Tokenization is a symmetric Treebank approximation, not exact Java PTB
tokenization. The exact training reward is therefore reported separately from
the standard evaluation CIDEr score.

NLTK METEOR remains available as an explicitly configured alternative
(`scst.reward = "nltk_meteor"`):

```python
meteor_score(
    [reference.casefold().split() for reference in references],
    prediction.casefold().split(),
)
```

Use the same token/text normalization for sampled and greedy captions. Validate
the required WordNet resource explicitly; do not download it as an import side
effect.

If training uses NLTK METEOR while the standardized report uses a COCO METEOR
implementation, report both with unambiguous names, for example
`reward_meteor_nltk` and `meteor_coco`. They are not interchangeable. The
primary objective-alignment comparison must include the exact training reward
implementation on validation/test predictions.

CIDEr-D optimization remains a different experiment from NLTK METEOR
optimization. Begin from an XE checkpoint when changing rewards; do not resume
an SCST checkpoint with a different reward protocol.

## Generation protocol

All decoders receive identical preprocessed images and return batched tensors.
Generation limits are expressed as `max_new_tokens`, separately from whether
BOS is stored in the returned sequence.

### Greedy

At each active step choose the highest-probability permitted token. Stop after
all examples emit EOS or reach the generation limit.

### Sampling

Sample categorically from permitted logits. Store the log probability of each
sampled action. Include the first EOS log probability and mask every forced
post-EOS action.

### Beam search

Maintain raw cumulative sequence log probability:

\[
\log P(y\mid x)=\sum_t \log p(y_t\mid y_{<t},x).
\]

Use the declared search-only Google NMT length penalty:

\[
lp(T)=\left(\frac{5+T}{6}\right)^\alpha, \qquad
score_{search}(y)=\frac{\log P(y\mid x)}{lp(T)}.
\]

The normalized search score is not the raw sequence log probability. Record
beam width, \(\alpha\), maximum new tokens, and special-token constraints.

## Evaluation

Generate exactly one prediction per ordered test image and store it keyed by
image ID. Every metric receives all references for that image.

Produce this complete matrix in one evaluation job:

| training checkpoint | greedy, multi-reference | beam-3, multi-reference |
|---|---:|---:|
| selected XE | metrics | metrics |
| selected SCST | metrics | metrics |

The evaluator must reject or visibly separate cells with different split,
reference, preprocessing, tokenizer, maximum-length, or metric identities.

Report:

- the exact training reward metric;
- corpus BLEU-1 through BLEU-4;
- standardized METEOR;
- ROUGE-L;
- CIDEr; and
- optionally SPICE when its external requirements are available.

Use a pinned COCO-compatible backend for the standardized suite. Do not average
sentence BLEU values. Metric implementation/version and tokenization are part
of the reported protocol.

## Checkpoint selection and test usage

- XE best: minimum validation token cross entropy.
- SCST best: maximum validation multi-reference greedy training reward.
- Test: evaluated only after selection; never used for early stopping,
  hyperparameter choice, or manual checkpoint choice.

Report both `best` and `last` only if their roles are explicit. The primary
matrix uses the predeclared selected checkpoints.

## Statistical reporting

For a descriptive project run, report per-image paired bootstrap confidence
intervals for XE-versus-SCST metric differences. For a claim that SCST reliably
improves the method, run multiple training seeds and report mean, spread, and
individual seed values. Never infer an SCST improvement by comparing
single-reference greedy XE with multi-reference beam SCST.

## Required result record

Every published result row must identify:

```text
run_id
training_stage and checkpoint_sha256
seed
split_manifest_sha256, reference_sha256, and image_files_sha256
resolved_model/data/reward config
decoder and decoder parameters
metric names/backends/versions
code revision and dirty status
environment/device identity
prediction artifact sha256
```

A missing field is reported as `unknown`; it is never reconstructed from a
README after the fact.
