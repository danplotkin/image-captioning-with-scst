# Checkpoint compatibility and provenance

## Compatibility goal

The notebook saved `model.state_dict()` directly. The new code supports those
raw tensor mappings for weight initialization/evaluation while using a
versioned envelope for new training checkpoints.

Weight compatibility is narrower than experiment reproducibility. Successfully
loading a historical state dictionary does not recover its split, code
execution order, dropout state, optimizer, RNG state, checkpoint-selection
rule, or evaluation protocol.

No historical checkpoint is currently checked into this repository. The
compatibility path can be unit-tested using synthetic legacy-shaped states, but
real parity remains pending until `cptr.pt` and/or `cptr_scst.pt` is supplied.

## Legacy architecture preset

A strict legacy load must construct the model with the notebook's surviving
architecture evidence:

| Property | Legacy value |
|---|---|
| encoder | `google/vit-base-patch16-384`, no pooling layer |
| tokenizer | `distilbert-base-uncased` |
| vocabulary | tokenizer base vocabulary; no added tokens |
| pad ID | `0` |
| maximum positional capacity | `80` |
| decoder model width | `768` |
| attention heads | `12` |
| feed-forward width | `1536` |
| decoder layers | `4` |
| decoder/positional dropout | `0.1` |
| activation | ReLU |
| decoder layout | `batch_first=True` |
| encoder policy | frozen |

Pin Hugging Face model/tokenizer revisions when they become known. The model
identifier alone is weaker than a revision or artifact hash.

## State-dict path contract

To avoid a destructive migration, new model construction retains these legacy
paths:

```text
backbone.backbone.*
positional_embedding.embedding.embedding.weight
positional_embedding.positional_encoding.pe
decoder.layers.*
fc.weight
fc.bias
```

Moving the decoder and ViT construction inside the model changes ownership but
does not require renaming those keys. With encoder width equal to decoder width,
an identity memory projection has no parameters and therefore introduces no
new state keys.

Strict load is the default. Missing or unexpected keys and shape mismatches are
errors, not warnings to suppress. Prefix rewriting such as removing
`module.` should only be performed by an explicit, logged conversion rule.

## What can and cannot be inferred

Some legacy configuration is visible in tensor shapes:

- vocabulary size from token embedding or output projection;
- model and feed-forward widths;
- decoder layer count; and
- positional capacity from the sinusoidal buffer.

Other facts cannot be recovered reliably from weights:

- number of attention heads, because projection weight shapes do not encode the
  head partition;
- exact encoder/tokenizer repository revision;
- data normalization, filters, or split IDs;
- dropout/model mode used during generation;
- optimizer, scheduler, scaler, epoch, or RNG state;
- checkpoint selection criterion; and
- whether a raw file is XE or SCST.

Consequently, a raw file requires the named legacy preset and records unknown
lineage as `unknown`.

## Safe loading

Load checkpoints on CPU by default and request tensor-safe loading where the
installed PyTorch version supports it:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

Then distinguish:

- a nonempty string-to-tensor mapping: schema 0 raw legacy state; or
- a mapping containing the supported `schema_version` and `model_state`: a
  versioned checkpoint.

Do not load untrusted pickle-based artifacts with arbitrary-object unpickling.
Calculate SHA256 before conversion and after writing the converted artifact.

## New checkpoint envelope

New checkpoints should contain at least:

```text
schema_version
stage                         # xe, scst, or explicitly unknown
model_state
resolved config
progress.epoch/global_step
best_metric name/value/split
provenance
```

Training checkpoints additionally include available optimizer, scheduler,
gradient-scaler, and RNG state. Provenance should include code revision/dirty
state, environment, encoder/tokenizer revisions, source/split/preprocessing
hashes, seed, and parent-checkpoint hash. Writes must be atomic and model state
should be moved to CPU before serialization.

Store `best` and `last` separately. A “best” filename without the metric name,
direction, split, and value in metadata is insufficient.

## Conversion workflow

The conversion command should:

1. calculate and record the input SHA256;
2. load the raw mapping safely on CPU;
3. construct the exact legacy preset;
4. strict-load every tensor;
5. record the stage as user-declared or `unknown`;
6. attach only provenance the user can substantiate;
7. run structural and, when possible, fixed-input parity validation;
8. write a versioned envelope atomically; and
9. calculate/output the converted SHA256 and validation report.

Conversion must not guess an original split or relabel an unknown file as the
best XE/SCST checkpoint based on its filename alone.

## Parity acceptance gate

Real historical compatibility is accepted only when all of the following pass:

1. zero missing/unexpected keys under strict load;
2. every tensor has the expected shape/dtype;
3. special IDs and tokenizer vocabulary match the output projection;
4. an independently implemented faithful legacy model and the new model,
   loaded with the same state and placed in eval mode, produce fixed-input
   logits equal within a declared tolerance;
5. greedy tokens match on a fixed image batch; and
6. the validation report records PyTorch/Transformers versions and artifact
   hashes.

Synthetic round-trip tests validate loader mechanics but do not satisfy this
real-artifact gate.

## Resume semantics

A raw notebook state dictionary can initialize XE, initialize corrected SCST,
or be evaluated. It cannot exactly resume training because optimizer,
scheduler, scaler, epoch/global step, and RNG state are absent.

The historical SCST weights also embody the train-mode greedy-baseline bug.
Loading them into an independent eval-mode model does not undo that training
history. The corrected SCST experiment must begin from an XE checkpoint and
form a new lineage.

## Shared-module contamination

The notebook assigned one global decoder and one global ViT object to every
`CPTR()` instance. A saved state dictionary still contains a coherent snapshot
of tensor values, so it may be technically loadable. However, the notebook's
nominal “new” model constructions were not independent, and a later load into
one instance could mutate the decoder observed by another instance.

For that reason, even a parity-valid legacy checkpoint does not validate the
historical before/after comparison. Historical scores remain unverified; new
claims require independently owned models and the experiment protocol in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).
