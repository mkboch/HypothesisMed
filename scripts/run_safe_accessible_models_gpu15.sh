#!/usr/bin/env bash
set -u

mkdir -p logs results/safe_runs

MODELS=(
  qwen2_5_7b_instruct
  deepseek_r1_qwen_32b
  qwen3_30b_thinking
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
    echo "RUNNING: ${model} | ${method}"
    echo "===================================================="

    CUDA_VISIBLE_DEVICES=1,5 PYTHONPATH=. python experiments/run_experiment.py \
      --model "$model" \
      --method "$method" \
      --max_samples 1000 \
      --batch_size 2 \
      2>&1 | tee "logs/${model}_${method}_safe_gpu15.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
      echo "FAILED: ${model}_${method}; skipping evaluation."
      continue
    fi

    src="results/${model}_${method}.jsonl"
    dst="results/safe_runs/${model}_${method}_gpu15_fixed.jsonl"

    if [ -s "$src" ]; then
      cp "$src" "$dst"
      echo "Saved safe copy: $dst"

      PYTHONPATH=. python experiments/evaluate_results.py \
        --path "$dst" \
        2>&1 | tee "logs/${model}_${method}_safe_gpu15_eval.log"
    else
      echo "WARNING: result file missing or empty: $src"
    fi
  done
done

echo "================ FINAL SAFE SUMMARY ================"
for file in results/safe_runs/*_gpu15_fixed.jsonl; do
  echo "----- $file -----"
  PYTHONPATH=. python experiments/evaluate_results.py --path "$file" || true
done
