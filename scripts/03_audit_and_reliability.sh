#!/usr/bin/env bash
set -e
source .venv/bin/activate 2>/dev/null || true

echo "===== Audit current Direct baseline ====="
PYTHONPATH=. python experiments/audit_results.py \
  --path results/qwen3_30b_thinking_direct.jsonl \
  --n 20

echo "===== Current metric summary ====="
for method in direct cot self_correction hypothesismed hypothesismed_nospace; do
  echo "----- $method -----"
  PYTHONPATH=. python experiments/evaluate_results.py \
    --path results/qwen3_30b_thinking_${method}.jsonl
done

echo "===== Reliability / calibration summary ====="
PYTHONPATH=. python experiments/reliability_analysis.py \
  --paths \
  results/qwen3_30b_thinking_direct.jsonl \
  results/qwen3_30b_thinking_cot.jsonl \
  results/qwen3_30b_thinking_self_correction.jsonl \
  results/qwen3_30b_thinking_hypothesismed.jsonl \
  results/qwen3_30b_thinking_hypothesismed_nospace.jsonl
