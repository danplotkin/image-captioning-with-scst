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

The training autocast context now sets `cache_enabled=False`. This disables the
**autocast weight cache**, not the attention KV cache. Incremental generation
remains available. A regression test checks gradients and Adam state for every
trainable tensor with both frozen and trainable image encoders. The existing AMP
overflow tests also exercise the corrected path on the original PyTorch runtime.

This fix changes future training behavior. It does not change existing weights,
repair historical training, or establish how a corrected SCST run would perform.
The originally reported lower SCST score cannot be attributed confidently to the
algorithm or its learning-rate schedule.

## Reproduce the audit

The historical checkpoints have been deleted. For a future audit, adapt the
script paths to available checkpoints, with the dataset prepared and a
CUDA environment matching `configs/evaluation-constraints.txt`:

```bash
HF_HOME=/workspace/.cache/huggingface \
  .venv/bin/python scripts/audit_scst_gradients.py
```

The script explicitly compares the historical and corrected autocast behavior
for greedy-baseline and zero-baseline rollouts, then checks saved parameter
equality against XE. It writes an audit JSON without saving modified models.

For a future controlled method comparison, initialize new RL runs from the same
XE parent using the corrected code and a declared validation protocol. The previous test results were deleted; prior exposure to the test set still
means that replacement runs must be tuned using validation only.
