#!/usr/bin/env bash
set -u
cd ~/HypothesisMed || exit 1
source /home/manikm/HypothesisMed/.venv/bin/activate
export PYTHONPATH=.
mkdir -p results/prof_revision_final logs

echo "================================================================================"
echo "START PROF REVISION REMAINING RUN"
date
echo "Host: $(hostname)"
echo "================================================================================"

DATA="results/prof_revision_space_stress/large_space_stress_inputs.jsonl"
PHI_OUT="results/microsoft_phi_4_mini_instruct_hypothesismed_v3_large_space_stress_inputs.jsonl"
MAX_SAMPLES=$(wc -l < "$DATA")

echo
echo "===== GPU status before Phi stress ====="
nvidia-smi

FREE_GPU=$(python - <<'PY'
import subprocess
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
pairs = []
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    pairs.append((int(mem), int(idx)))
pairs.sort()
print(pairs[0][1])
PY
)

echo "Using GPU: $FREE_GPU"
echo "Stress data: $DATA"
echo "Max samples: $MAX_SAMPLES"

if [ -f "$PHI_OUT" ] && [ "$(wc -l < "$PHI_OUT")" -ge "$MAX_SAMPLES" ]; then
  echo "[SKIP] Phi stress output already complete: $PHI_OUT"
else
  CUDA_VISIBLE_DEVICES=$FREE_GPU python scripts/run_local_cached_model_experiment.py \
    --model_name "microsoft/Phi-4-mini-instruct" \
    --model_path "microsoft/Phi-4-mini-instruct" \
    --method hypothesismed_v3 \
    --data "$DATA" \
    --max_samples "$MAX_SAMPLES" \
    --batch_size 16 \
    --max_model_len 4096 \
    --max_tokens 512
fi

echo
echo "===== Run final analysis ====="
python scripts/prof_revision_final_analysis.py

echo
echo "================================================================================"
echo "DONE PROF REVISION REMAINING RUN"
date
echo "================================================================================"
