#!/usr/bin/env bash
set -e
source .venv/bin/activate

MODELS=(
  qwen3_235b_thinking
  qwen3_235b_instruct
  deepseek_r1_0528
  glm_5_1
  glm_5
)

for model in "${MODELS[@]}"; do
  echo "Running large model: $model"
  PYTHONPATH=. python experiments/run_experiment.py \
    --model "$model" \
    --method hypothesismed \
    --max_samples 200 \
    --batch_size 2

  PYTHONPATH=. python experiments/evaluate_results.py \
    --path "results/${model}_hypothesismed.jsonl" \
    > "results/${model}_hypothesismed_metrics.json"
done
