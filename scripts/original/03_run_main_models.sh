#!/usr/bin/env bash
set -e
source .venv/bin/activate

MODELS=(
  biomistral_7b
  qwen3_30b_thinking
  gpt_oss_120b
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
    echo "Running $model / $method"
    PYTHONPATH=. python experiments/run_experiment.py \
      --model "$model" \
      --method "$method" \
      --max_samples 200 \
      --batch_size 4

    PYTHONPATH=. python experiments/evaluate_results.py \
      --path "results/${model}_${method}.jsonl" \
      > "results/${model}_${method}_metrics.json"
  done
done
