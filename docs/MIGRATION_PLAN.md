# Notebook-to-codebase migration plan

## Objective

Turn the Colab proof of concept into a testable Python project that can:

1. load historical notebook weights where possible;
2. train the CPTR-inspired model with cross entropy;
3. fine-tune it with a faithful, multi-reference SCST procedure;
4. evaluate XE and SCST checkpoints under identical greedy and beam protocols;
5. preserve enough provenance to reproduce or audit every reported number; and
6. keep notebooks as optional clients of the package rather than the source of
   truth.

Correct scientific behavior and notebook compatibility are separate concerns.
The project separates two experiment lineages:

- `legacy_notebook`: strict raw-weight conversion and historical inspection.
  It is deliberately not an executable training profile because the original
  split, checkpoints, and environment are absent; inventing them would create
  false provenance.
- `flickr8k_scst`: uses the corrected image-level, multi-reference,
  inference-mode SCST and standardized evaluation protocol.

No corrected run should be described as numerically reproducing the notebook.
Corrections to dropout, references, transforms, action constraints, and
evaluation intentionally alter the computation.

## Target boundaries

The package should keep the following responsibilities independent:

```text
src/scst_captioner/
  config.py           typed, validated experiment configuration
  reproducibility.py  seeds, worker setup, environment capture
  data.py or data/    parsing, records, manifests, datasets, transforms
  model.py            independently owned CPTR-inspired modules
  generation.py       greedy, sampling, beam and rollout outputs
  losses.py           token XE and explicit-mask REINFORCE loss
  rewards.py          multi-reference reward interface and implementations
  checkpointing.py    legacy loader and versioned training checkpoints
  train_xe.py         supervised engine
  train_scst.py       SCST engine
  evaluation.py       fixed-protocol prediction and metric matrix
  cli.py              side-effect-free command entry point
```

Configuration belongs in versioned TOML files. Data, checkpoints, predictions,
and runs belong under ignored runtime directories, not in package modules. A
run directory should contain the resolved config, logs/history, environment
metadata, best and last checkpoints, predictions, metrics, and hashes.

Recommended CLI workflow:

```text
scst-captioner prepare-data
scst-captioner validate-data
scst-captioner train-xe
scst-captioner train-scst
scst-captioner evaluate
scst-captioner evaluate-matrix
scst-captioner caption
scst-captioner convert-checkpoint
```

Imports and `--help` must not mount Drive, read Colab secrets, download NLTK or
Hugging Face assets, or require a GPU.

## Phases and acceptance gates

Each phase is complete only when its gate passes. Later phases may be developed
in parallel, but experimental claims must respect the gate order.

Implementation status after this migration:

| Gate | Status |
|---|---|
| 0: audit/protocol | Implemented and statically validated |
| 1: package/config/reproducibility | Implemented; CI covers Python 3.11–3.13 |
| 2: data/manifests | Implemented with synthetic tests; the real manifest awaits Flickr8K |
| 3: model/checkpoints | Implemented with synthetic strict round trips; historical weights are absent |
| 4: generation/SCST | Implemented with analytic, mode, shape, and EOS tests |
| 5: training/evaluation | Implemented and hermetically tested; the Java COCO suite cannot run on this host |
| 6: real experiment | Pending external data, fresh compute, checkpoints, and result artifacts |

### Phase 0: freeze the audit and protocol

Deliverables:

- implementation audit with issue IDs and severities;
- this migration plan;
- experiment protocol and research-alignment documents;
- checkpoint compatibility contract; and
- an explicit historical/unverified label on old scores.

Gate 0:

- every substantive notebook definition has a destination module;
- legacy and corrected behavior are distinguishable in configuration;
- historical scores are not copied into a new-results table; and
- known unknowns—checkpoint files, original manifest, dependency snapshots—are
  recorded instead of inferred.

### Phase 1: package, configuration, and reproducibility

Deliverables:

- installable `src` package and console entry point;
- typed TOML configuration with strict unknown-key validation;
- direct runtime and development dependencies;
- seed/device/AMP utilities and environment capture;
- root ignore policy for datasets, checkpoints, generated runs, caches, and
  local environments; and
- CPU CI for lint and tests.

Gate 1:

- `pip install -e .` succeeds in a clean environment;
- the package imports and CLI help run offline without side effects;
- invalid model dimensions, split paths, reward names, and decoder parameters
  fail early with useful errors;
- Python, NumPy, Torch, CUDA, DataLoader workers, and DataLoader generators are
  seeded from the resolved run seed; and
- a smoke config is small enough for routine CPU CI.

### Phase 2: canonical data and split manifests

Deliverables:

- robust caption CSV parser preserving raw and normalized references;
- canonical image record containing image ID/path and all references;
- caption-row dataset view for XE;
- image-reference dataset view for SCST/evaluation;
- PIL-first training augmentation and deterministic evaluation preprocessing;
- stable split generator and versioned manifest; and
- dataset/source/preprocessing hashes.

Gate 2:

- train, validation, and test image sets are pairwise disjoint;
- the union of train, validation, test, and explicitly excluded IDs equals the
  canonical image set, and excluded IDs are never used;
- regenerating a manifest with the same source, algorithm, and seed is byte
  stable;
- every SCST/evaluation item appears exactly once and carries all references;
- XE contains the expected image/reference pairs;
- batch size one and variable-length reference collation work; and
- once Flickr8K is available, the legacy profile reproduces the notebook's
  saved 7,467 image and 23,890/5,975/7,470 caption-row counts or records a
  concrete, investigated discrepancy.

The corrected default should retain all valid nonempty references unless an
explicit dataset policy says otherwise. It should not inherit the notebook's
incidental “drop the entire image if any caption is short” behavior.

### Phase 3: model ownership and checkpoint compatibility

Deliverables:

- model instances that own their ViT and Transformer decoder;
- frozen encoder kept in eval mode and evaluated without gradients;
- legacy-compatible parameter paths and strict raw-state loading;
- versioned, atomic checkpoints with resolved config/provenance; and
- a conversion/inspection command for legacy files.

Gate 3:

- two newly constructed models do not share encoder or decoder parameter
  storage;
- teacher-forced forward, causal mask, padding mask, and positional-capacity
  tests pass with a tiny injected encoder;
- raw legacy-shaped state dictionaries round-trip with no missing or unexpected
  keys under the legacy preset;
- checkpoint save/load is map-location aware and defaults to safe loading; and
- if historical checkpoints are supplied, eval-mode fixed-input logits match a
  faithful legacy model within documented numeric tolerances.

Loading a raw checkpoint does not validate the experiment that produced it. It
only validates weight compatibility.

### Phase 4: shape-stable generation and SCST

Deliverables:

- a generation result retaining `[batch, time]` for every batch size;
- explicit action mask including first EOS and excluding post-EOS steps;
- shared special-token constraints across sample, greedy, and beam decoding;
- declared Google-NMT-style beam length penalty distinct from raw log
  probability;
- multi-reference reward abstraction;
- inference-mode greedy baseline; and
- SCST engine with detached advantage, count-weighted statistics, gradient
  clipping, and device-aware AMP.

Gate 4:

- batch sizes one and greater than one pass sample/greedy/beam tests;
- EOS and post-EOS masks pass exact tensor tests;
- a hand-computed example verifies the REINFORCE loss value and gradient sign;
- the baseline has no gradient and observes `training == False`;
- the corrected sampled branch observes `training == False` but retains policy
  gradients;
- sampled and baseline reward calls receive the identical complete reference
  list for each image;
- non-EOS special tokens cannot be sampled or selected; and
- a tiny offline SCST step produces finite loss and expected parameter
  gradients.

An optional train-mode sampled policy can exist only as an explicitly named
ablation. It is not the corrected default.

### Phase 5: supervised training and fixed-protocol evaluation

Deliverables:

- XE trainer with summed/count-weighted validation and best/last checkpoints;
- SCST checkpoint selection using validation-only, multi-reference greedy
  reward;
- ordered prediction artifacts keyed by image ID;
- same-reward evaluation plus a pinned COCO-compatible corpus evaluator; and
- automatic XE/SCST by greedy/beam matrix.

Gate 5:

- a tiny offline CPU integration run completes an XE update, an SCST update,
  best/last checkpoint loading, and greedy/beam evaluation;
- all four matrix cells share split, reference, preprocessing, tokenizer, and
  metric identities;
- corpus BLEU is used; no code averages sentence BLEU for research results;
- metric denominators count tokens or images, not batches;
- test data is never used for checkpoint selection; and
- every result artifact includes resolved config, code revision/dirty state,
  environment, seed, split hash, checkpoint hash, decoding settings, and metric
  implementation versions.

### Phase 6: real-data validation and notebook retirement

Deliverables:

- fresh XE and corrected SCST runs from a clean environment;
- four-way test matrix with paired image IDs;
- paired bootstrap confidence intervals and, for a strong improvement claim,
  repeated training seeds;
- README/model-card update generated from validated result artifacts; and
- original notebook archived or reduced to a thin package demo.

Gate 6:

- a clean-clone procedure can regenerate the same split manifest and complete
  the smoke run;
- real runs pass all invariant checks and retain best/last checkpoints plus
  predictions;
- the primary SCST comparison uses identical greedy, multi-reference
  evaluation for XE and SCST;
- beam results are reported as a separate decoder factor;
- old scores remain historical/unverified; and
- project language says “CPTR-inspired” and identifies the frozen encoder and
  METEOR-reward adaptations.

## Test strategy

Unit tests should avoid network and large pretrained models. Inject a tiny
vision encoder and tokenizer-compatible special IDs. At minimum cover:

- robust comma/quote parsing and caption normalization;
- split determinism, leakage, coverage, and manifest hashing;
- reference-aware collation and batch-one behavior;
- model-instance independence and state-dict key compatibility;
- causal/padding masks and maximum positional length;
- generation shape, EOS mask, forbidden-token mask, and beam score;
- exact REINFORCE loss/sign/detachment;
- baseline/sample modes and gradient boundary;
- multi-reference METEOR adapter behavior;
- count-weighted XE/reward aggregation; and
- raw and versioned checkpoint round trips.

Integration tests should run a two- or three-image synthetic pipeline entirely
on CPU. Full ViT/Flickr8K tests should be marked separately because they require
large external artifacts.

## Completion definition

The code migration is complete when Gates 0--5 pass. The experimental project
is complete only after Gate 6 produces new, provenance-rich results. Passing
unit tests without the real dataset is implementation validation, not a
reproduction of the historical experiment.
