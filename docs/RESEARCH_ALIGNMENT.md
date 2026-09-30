# Research alignment and claim boundaries

## Research basis

The implementation is informed by, but does not exactly reproduce, the
following work:

1. Rennie et al., [Self-Critical Sequence Training for Image
   Captioning](https://arxiv.org/abs/1612.00563), arXiv 2016 / CVPR 2017.
2. Liu et al., [CPTR: Full Transformer Network for Image
   Captioning](https://arxiv.org/abs/2101.10804), 2021.
3. Vaswani et al., [Attention Is All You
   Need](https://arxiv.org/abs/1706.03762), 2017.
4. Banerjee and Lavie, [METEOR: An Automatic Metric for MT Evaluation with
   Improved Correlation with Human
   Judgments](https://aclanthology.org/W05-0909/), 2005, and Lavie and Agarwal,
   [METEOR: An Automatic Metric for MT Evaluation with High Levels of
   Correlation with Human Judgments](https://aclanthology.org/W07-0734/), 2007.
5. Chen et al., [Microsoft COCO Captions: Data Collection and Evaluation
   Server](https://arxiv.org/abs/1504.00325), 2015, for the standardized image
   caption evaluation setting.
6. Vedantam et al., [CIDEr: Consensus-based Image Description
   Evaluation](https://arxiv.org/abs/1411.5726), 2015.

Paper dates should distinguish an arXiv preprint from its later conference
publication when relevant.

## Alignment matrix

| Research element | Project alignment | Project deviation/decision |
|---|---|---|
| Transformer caption decoder | Uses causal self-attention over caption tokens and cross-attention to visual memory. | Uses PyTorch's standard `TransformerDecoder` and a DistilBERT tokenizer rather than claiming an exact paper implementation. |
| CPTR-style full-Transformer captioning | Uses a ViT representation as memory for a Transformer language decoder. | Uses a pretrained Hugging Face ViT and freezes it by default. Model construction, data, and training recipe differ from an exact CPTR reproduction. |
| REINFORCE sequence objective | Uses sampled sequence log probability multiplied by a sequence-level advantage with the correct negative sign. | Uses an explicit first-EOS action mask and implementation-specific batching. |
| Self-critical baseline | Uses reward of the model's own greedy test-time output as the baseline. | The historical notebook incorrectly left dropout active. The corrected protocol runs greedy baseline and sampled policy in eval mode while retaining autograd for sampled log probabilities. |
| Original SCST reward | SCST supports nondifferentiable sequence-level rewards. | The default is multi-reference CIDEr-D, with project-specific tokenization and cached training-corpus document frequencies. METEOR is an optional alternative. |
| Multiple human references | Both sampled and baseline captions are scored against the complete reference set for the image. | Historical notebook SCST used one reference at a time and is retained only as legacy behavior. |
| Test-time inference baseline | Greedy baseline is the declared default inference approximation. | Beam-3 is also evaluated. It is a separate decoder factor and does not retroactively make the greedy-baseline training invalid. |
| Caption evaluation | Uses one fixed image list, all references, and corpus/COCO-compatible metrics. | Historical mean sentence BLEU and mixed-protocol tables are not research-comparable. |

## Why eval-mode sampling is the corrected default

The defining self-critical reference is the score of the model under its own
test-time inference algorithm. `torch.no_grad()` only controls graph recording;
it does not change dropout or other training-mode behavior. Therefore the
greedy reference must run with `model.eval()`.

For this project, the sampled branch also runs in eval mode. Evaluation mode
does not disable autograd, so sampled-token log probabilities still receive
gradients. This choice makes the sampled policy and greedy reference use the
same inference-time network and avoids adding dropout noise to the advantage.
Train-mode sampling can be studied as an explicitly labeled ablation, but it
must not be confused with the corrected default.

## Reward alignment

SCST is not inherently restricted to CIDEr. Its estimator only requires a
sequence-level reward that can be evaluated on sampled and baseline captions.
The default is multi-reference CIDEr-D; METEOR is an optional project objective.

The reward choice has two implications:

1. Results should name the configured reward (CIDEr-D by default, or NLTK
   METEOR when explicitly selected), without implying exact paper reproduction.
2. The exact training reward implementation should be applied to evaluation
   predictions and reported alongside standardized COCO metrics. The cached
   training CIDEr-D and evaluation CIDEr use different corpus/tokenization
   protocols; NLTK and COCO METEOR also require distinct names.

The reward must receive all references. A single reference can penalize a
semantically valid caption merely because it chooses a different human
phrasing; it also mismatches a final multi-reference objective.

## Architecture claim

The safe description is:

> A CPTR-inspired image captioner using a frozen pretrained ViT encoder and an
> autoregressive Transformer decoder, first trained with cross entropy and then
> fine-tuned with greedy-baseline SCST using multi-reference CIDEr-D reward.

Avoid these stronger descriptions unless a separate verified implementation
supports them:

- “exact CPTR reproduction”;
- “reproduces the CPTR paper score”;
- “reproduces the original SCST experiment”; or
- “state of the art.”

Freezing the pretrained encoder is a reasonable compute-saving design choice,
but it changes optimization and should be stated prominently.

## Evaluation alignment

The training-stage question and decoder question form a two-by-two design:

| checkpoint | greedy | beam-3 |
|---|---:|---:|
| XE | same references/metrics | same references/metrics |
| SCST | same references/metrics | same references/metrics |

The XE-to-SCST difference within the greedy column is the primary SCST
comparison. The greedy-to-beam difference within a row is the decoder
comparison. Comparing XE greedy/single-reference to SCST beam/multi-reference
changes three variables and supports no isolated claim.

Caption metrics capture different aspects and are imperfect proxies for human
judgment. Research-facing reports should therefore use a standard suite rather
than optimize the presentation around one favorable number. At minimum report
corpus BLEU-1--4, METEOR, ROUGE-L, and CIDEr, with SPICE optional because of its
heavier external requirements.

## Reproducibility versus replication

- **Implementation reproducibility** means a clean environment can run the
  package/tests and regenerate a declared split and smoke result.
- **Experimental reproducibility** means a run carries enough configuration,
  data/checkpoint identity, predictions, and environment information to repeat
  its procedure.
- **Paper replication** means matching the paper's architecture, data,
  training, and evaluation protocol closely enough to compare outcomes.

This project targets the first two. It does not currently claim the third for
CPTR or the original CIDEr-D SCST experiment.

## Historical evidence policy

The notebook's saved scores are useful historical context but are unverified:

- cells were executed nonlinearly;
- model instances shared a global decoder and encoder;
- the exact split was not persisted;
- checkpoints and environment lock are absent;
- training/evaluation used inconsistent reference and decoding protocols; and
- current README values disagree with current notebook output/configuration.

These scores must be labeled “historical notebook output; not reproduced under
the current protocol.” They are not regression thresholds. New results begin a
new lineage only after the acceptance gates in
[MIGRATION_PLAN.md](MIGRATION_PLAN.md) pass.

## Claim checklist

Before publishing a number or conclusion, verify:

- [ ] the checkpoint SHA256 and parent lineage are recorded;
- [ ] the exact ordered split/reference hashes are recorded;
- [ ] sampled and baseline SCST rewards used all references;
- [ ] baseline generation ran in eval mode;
- [ ] the compared result cells share preprocessing, decoder, and metrics;
- [ ] corpus rather than mean sentence BLEU is reported;
- [ ] metric implementation/version is named;
- [ ] seed, code revision, environment, and prediction artifact are recorded;
- [ ] test data was not used for checkpoint selection; and
- [ ] the language says CPTR-inspired and names the configured reward and
      frozen-ViT adaptation.
