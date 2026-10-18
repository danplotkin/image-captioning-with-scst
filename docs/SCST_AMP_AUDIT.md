# SCST mixed-precision training audit

Final-evaluation preparation on 2026-10-18 uncovered a training defect in the
historical SCST run. Its saved scores describe real checkpoints, but the run is
**not a valid matched comparison of SCST versus vanilla REINFORCE**.

## Evidence from the actual checkpoint

The CUDA audit used the saved best SCST checkpoint (epoch 17), its pinned ViT and
tokenizer, and the first validation image. No optimizer updates were performed.

| Check | Result |
| --- | ---: |
| Trainable parameter tensors | 141 |
| Saved SCST Adam states | 99 |
| Saved REINFORCE Adam states | 141 |
| SCST tensors missing gradients with the historical autocast weight cache | 42 |
| Those 42 tensors still exactly equal to the XE parent | 42 |
| Corresponding tensors changed in REINFORCE | 42 |
| Missing gradients with the autocast weight cache disabled | 0 |

The 42 tensors are the weights and biases of the self-attention input/output
projections, cross-attention output projections, and two feed-forward linears in
each of four decoder layers, plus the vocabulary output head. Other trainable
parameters did receive updates, so a finite loss and a nonzero optimizer step
counter were insufficient checks.

The observations above were recorded before the experiment reset. At the user's
request, both historical RL runs and all evaluation/audit output files were
deleted. The checkpoint-based audit can no longer be reproduced from those
deleted artifacts; this document preserves the defect explanation, not results.

## Cause and correction

In PyTorch 2.8.0+cu128, the greedy baseline runs under `no_grad()` inside the same
autocast context as the subsequent differentiable sample. Autocast can cache
detached FP16 casts of leaf weights during that baseline and reuse them during
sampling. The sample still has some differentiable paths, so training can proceed
while those weights silently receive no gradients. REINFORCE skips the greedy
baseline and did not show the missing-gradient behavior in the matched audit.

The first fix disabled autocast weight caching. The current implementation
instead gives the shared encoder, no-grad baseline, and differentiable sampled
rollout separate, non-overlapping autocast scopes with `cache_enabled=True`.
Exiting the baseline's outermost scope clears its detached weight casts before
sampling. The trainer does not create a shared outer autocast scope;
`scst_rollouts(..., amp=True)` owns the scopes and rejects an active outer scope.
A trainable shared encoder retains its gradient graph; only the baseline gets
detached image features. Train-mode sampling still recomputes a trainable encoder
to preserve the different dropout modes. REINFORCE skips the baseline entirely.

This follows the [PyTorch autocast maintainer's explanation and recommended
separate scopes](https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475).
The attention KV cache is independent and remains supported.
Regression tests check finite gradients and Adam state for every trainable tensor
for both methods, frozen/trainable encoders, train/eval sampling, KV caching
on/off, and CPU/CUDA FP16. Separate tests check non-overlapping scope ownership,
weight-cache enablement, the no-grad baseline, and differentiable FP16 sampling.

This fix changes future training behavior. It does not change existing weights,
repair historical training, or establish how a corrected SCST run would perform.
The originally reported lower SCST score cannot be attributed confidently to the
algorithm or its learning-rate schedule.

## Reproduce the audit

The audit uses the retained XE parent and the first validation image, without
optimizer updates or test-set evaluation. In a CUDA environment matching
`configs/evaluation-constraints.txt`, run:

```bash
HF_HOME=/workspace/.cache/huggingface \
  .venv/bin/python scripts/audit_scst_gradients.py --require-pod-runtime
.venv/bin/python -m pytest tests/test_amp.py -q
```

The script recreates the old shared-context ordering as a positive control. It
then verifies finite gradient coverage for both methods and compares sample
tokens and gradients with weight caching enabled/disabled across both sampling
modes and KV-cache settings. Results are saved to `verification/amp-gradients.json`;
no trained checkpoint is created or modified.

For a future controlled method comparison, initialize new RL runs from the same
XE parent using the corrected code and a declared validation protocol. The previous test results were deleted; prior exposure to the test set still
means that replacement runs must be tuned using validation only.

## Refactor verification (2026-10-18)

- Local suite: 301 passed; 16 CUDA-only cases skipped on the Mac.
- RTX 4090, PyTorch `2.8.0+cu128`, CUDA runtime `12.8`: 317 passed.
- Retained XE parent, first validation image, three sampled tokens: all 141
  trainable parameter tensors received finite gradients in all 16 audit cases.
- Historical shared-scope control reproduced 42 missing gradients.
- Separate scopes with weight caching on/off produced identical sampled tokens;
  gradients were close (`rtol=0.03`, `atol=0.002`) across both methods, sampling
  modes, and KV-cache settings. This is numerical agreement, not bitwise identity.
- No optimizer updates in the saved-checkpoint audit. The suite's tiny-model
  tests separately verify successful optimizer updates and Adam state coverage.
- No project training or test-set evaluation was launched. The temporary pod
  and its disk were deleted after verification records were downloaded.

The [audit JSON](../verification/amp-gradients.json),
[GPU test log](../verification/pod-tests.log),
[source hashes](../verification/source-sha256.json), and
[cleanup receipt](../verification/cloud-cleanup.json) record this check.
This short validation-image audit establishes gradient connectivity and numerical
agreement on the tested cases; it does not establish convergence or full-batch
training memory requirements.
