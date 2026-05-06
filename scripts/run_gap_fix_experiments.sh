#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"

mkdir -p results/reparsed results/summary logs

echo "Using GPUs: ${GPU_IDS}"
echo "Samples: ${MAX_SAMPLES}"
echo "Batch size: ${BATCH_SIZE}"

echo "===== 1. Re-evaluate existing Qwen3-30B results with fixed parser ====="
for method in direct cot self_correction hypothesismed hypothesismed_nospace; do
  if [ -f "results/reparsed/qwen3_30b_thinking_${method}.jsonl" ]; then
    echo "----- qwen3_30b_thinking / ${method} -----"
    PYTHONPATH=. python experiments/evaluate_results.py \
      --path "results/reparsed/qwen3_30b_thinking_${method}.jsonl"
  fi
done

echo "===== 2. Run second model: Qwen2.5-7B-Instruct ====="
for method in direct cot self_correction hypothesismed; do
  echo "----- qwen2_5_7b_instruct / ${method} -----"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python experiments/run_experiment.py \
    --model qwen2_5_7b_instruct \
    --method "${method}" \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee "logs/qwen2_5_7b_instruct_${method}.log"

  PYTHONPATH=. python experiments/evaluate_results.py \
    --path "results/qwen2_5_7b_instruct_${method}.jsonl"
done

echo "===== 3. Run second dataset if supported by --data ====="
for method in direct cot self_correction hypothesismed; do
  echo "----- qwen3_30b_thinking / ${method} / medqa -----"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python experiments/run_experiment.py \
    --model qwen3_30b_thinking \
    --method "${method}" \
    --data medqa \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee "logs/qwen3_30b_thinking_${method}_medqa.log"

  PYTHONPATH=. python experiments/evaluate_results.py \
    --path "results/qwen3_30b_thinking_${method}.jsonl"
done

echo "===== 4. Print all available summaries ====="
for f in results/*.jsonl results/reparsed/*.jsonl; do
  [ -f "$f" ] || continue
  echo "----- $f -----"
  PYTHONPATH=. python experiments/evaluate_results.py --path "$f" || true
done

echo "Done."
