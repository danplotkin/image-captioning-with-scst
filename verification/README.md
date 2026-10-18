# AMP scope verification

These are correctness checks, not training runs or test-set caption evaluations.
See [the audit explanation](../docs/SCST_AMP_AUDIT.md).

The full GPU suite and actual XE checkpoint audit ran on an RTX 4090 with
PyTorch 2.8.0+cu128 / CUDA 12.8. The historical shared scope reproduced 42 missing
gradients; the isolated scopes had none. No saved checkpoint was updated.
`source-sha256.json` was verified on the pod against the local implementation.
The temporary verification pod was deleted; see `cloud-cleanup.json`.
