#!/usr/bin/env bash
set -u

cd ~/HypothesisMed
source /home/manikm/HypothesisMed/.venv/bin/activate
export PYTHONPATH=.

echo "===== Creating additional datasets ====="
python scripts/create_multidataset_benchmark.py

DATASETS=()
for f in datasets/transformed/*original1000.jsonl; do
  [ -s "$f" ] || continue
  stem=$(basename "$f" .jsonl)
  DATASETS+=("$stem")
done

# Use Qwen as primary model. DeepSeek is included as secondary model, but if it remains unstable,
# use it as appendix/robustness analysis rather than main claim.
MODELS=("qwen2_5_7b_instruct" "deepseek_r1_qwen_32b")
METHODS=("direct" "cot" "hypothesismed_v3")

echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Methods: ${METHODS[*]}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '$2+0 > 60000 {gsub(/ /,"",$1); print $1; exit}'
}

for dataset in "${DATASETS[@]}"; do
  data_path="datasets/transformed/${dataset}.jsonl"

  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do

      expected="results/${model}_${method}_${dataset}.jsonl"
      if [ -s "$expected" ] && [ "$(wc -l < "$expected")" -ge 950 ]; then
        echo "===== SKIP existing complete: $expected ====="
        python scripts/reparse_results_inplace.py --glob "$expected" || true
        continue
      fi

      FREE_GPU=$(pick_gpu)
      while [ -z "$FREE_GPU" ]; do
        echo "No GPU with >60GB free. Sleeping 120 sec..."
        sleep 120
        FREE_GPU=$(pick_gpu)
      done

      export CUDA_VISIBLE_DEVICES=$FREE_GPU
      echo "=================================================="
      echo "RUNNING dataset=$dataset model=$model method=$method GPU=$FREE_GPU"
      echo "=================================================="

      python experiments/run_experiment.py \
        --model "$model" \
        --method "$method" \
        --data "$data_path" \
        --max_samples 1000 \
        --batch_size 1 || true

      python scripts/reparse_results_inplace.py --glob "results/${model}_${method}_${dataset}.jsonl" || true
      python scripts/create_multidataset_fusion_and_summary.py || true
    done
  done
done

echo "===== Final fusion and summary ====="
python scripts/reparse_results_inplace.py --glob "results/*original1000*.jsonl" || true
python scripts/create_multidataset_fusion_and_summary.py

echo "===== Final aggregate table ====="
cat results/final_multidataset/aggregate_results_across_datasets.csv

echo "===== Final per-dataset table ====="
cat results/final_multidataset/all_results_by_dataset_model_method.csv
