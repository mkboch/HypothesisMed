#!/usr/bin/env bash
set -e
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python experiments/run_experiment.py \
  --model qwen2_5_7b_instruct \
  --method hypothesismed \
  --max_samples 5 \
  --batch_size 1

PYTHONPATH=. python experiments/evaluate_results.py \
  --path results/qwen2_5_7b_instruct_hypothesismed.jsonl
