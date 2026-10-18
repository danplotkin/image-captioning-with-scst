#!/usr/bin/env bash
# On-pod execution after upload/install. Every failure leaves logs and exits nonzero.
set -euo pipefail
cd /workspace/image-captioning-with-scst
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p artifacts/final-evaluation/cache-validation
for method in xe reinforce scst; do
  case "$method" in
    xe) run=flickr30k-xe-finetune-runpod ;;
    *) run=flickr30k-policy-gradient/${method} ;;
  esac
  .venv/bin/python -u scripts/verify_kv_cache.py --checkpoint "artifacts/$run/best.pt" \
    --decoding beam --images 16 --device cuda \
    --output "artifacts/final-evaluation/cache-validation/$method-cuda-fp32.json"
done
amp_ok=true
for method in xe reinforce scst; do
  case "$method" in
    xe) run=flickr30k-xe-finetune-runpod ;;
    *) run=flickr30k-policy-gradient/${method} ;;
  esac
  if ! .venv/bin/python -u scripts/verify_kv_cache.py --checkpoint "artifacts/$run/best.pt" \
    --decoding beam --images 16 --device cuda --amp \
    --output "artifacts/final-evaluation/cache-validation/$method-cuda-fp16.json"; then
    amp_ok=false
  fi
done
args=()
if "$amp_ok"; then args+=(--amp); fi
printf 'All-method FP16 cache equivalence: %s\n' "$amp_ok"
.venv/bin/python -u scripts/evaluate_final_models.py --device cuda "${args[@]}"
touch artifacts/final-evaluation/EVALUATION_COMPLETE
