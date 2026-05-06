#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

MODELS=(
  qwen2_5_72b_instruct
  deepseek_r1_distill_llama_70b
  llama_3_1_8b_instruct
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
      2>&1 | tee "logs/${model}_${method}_1000.log"

    PYTHONPATH=. python experiments/evaluate_results.py \
      --path "results/${model}_${method}.jsonl" \
      2>&1 | tee "logs/${model}_${method}_1000_eval.log"
  done
done

echo "================ FINAL SUMMARY ================"
for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "----- ${model}_${method} -----"
    PYTHONPATH=. python experiments/evaluate_results.py \
      --path "results/${model}_${method}.jsonl" || true
  done
done
