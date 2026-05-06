#!/usr/bin/env bash
set -u

mkdir -p logs backups
cp configs/models.yaml backups/models_before_all_runnable_$(date +%Y%m%d_%H%M%S).yaml

python - <<'PY'
from pathlib import Path
import yaml

p = Path("configs/models.yaml")
cfg = yaml.safe_load(p.read_text())

new_models = {
    "qwen2_5_72b_instruct": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "tensor_parallel_size": 2
    },
    "deepseek_r1_distill_llama_70b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "tensor_parallel_size": 2
    },
    "llama_3_1_8b_instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 1
    },
    "llama_3_3_70b": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "tensor_parallel_size": 2
    },
    "meditron_70b": {
        "hf_id": "epfl-llm/meditron-70b",
        "tensor_parallel_size": 2
    }
}

if not isinstance(cfg, dict):
    raise TypeError("configs/models.yaml must be a dictionary")

cfg.update(new_models)
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Updated configs/models.yaml with missing runnable model keys.")
PY

MODELS=(
  qwen2_5_7b_instruct
  deepseek_r1_qwen_32b
  qwen3_30b_thinking
  qwen2_5_72b_instruct
  deepseek_r1_distill_llama_70b
  llama_3_1_8b_instruct
  llama_3_3_70b
  meditron_70b
)

METHODS=(
  direct
  cot
  self_correction
  hypothesismed
)

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "===================================================="
    echo "START: model=${model}, method=${method}"
    echo "===================================================="

    CUDA_VISIBLE_DEVICES=1,5 PYTHONPATH=. python experiments/run_experiment.py \
      --model "$model" \
      --method "$method" \
      --max_samples 1000 \
      --batch_size 2 \
      2>&1 | tee "logs/${model}_${method}_1000_gpu15.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
      echo "FAILED: model=${model}, method=${method}, status=${status}"
      echo "Continuing to next run..."
      continue
    fi

    PYTHONPATH=. python experiments/evaluate_results.py \
      --path "results/${model}_${method}.jsonl" \
      2>&1 | tee "logs/${model}_${method}_1000_gpu15_eval.log"
  done
done

echo "================ FINAL SUMMARY ================"
for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    file="results/${model}_${method}.jsonl"
    if [ -f "$file" ]; then
      echo "----- ${model}_${method} -----"
      PYTHONPATH=. python experiments/evaluate_results.py --path "$file" || true
    fi
  done
done
