#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/manikm/HypothesisMed

CLIENT_PY="/home/manikm/HypothesisMed/.venv/bin/python3"
if [ -x "/home/manikm/miniconda3/envs/vllm311/bin/python" ]; then
  VLLM_PY="/home/manikm/miniconda3/envs/vllm311/bin/python"
else
  VLLM_PY="$CLIENT_PY"
fi

RUNPY="scripts/expanded_medical_model_evaluation.py"
OUT="results/expanded_final_gap"
PHASE2="results/expanded_fix_vllm"
GPU_ID="5"
PORT_BASE="${PORT_BASE:-19350}"
QA_MAX_PER_DATASET="${QA_MAX_PER_DATASET:-1000}"
SPACE_MAX_ROWS="${SPACE_MAX_ROWS:-0}"

mkdir -p "$OUT/outputs" logs

CURRENT_SERVER_PGID=""

stop_server_group() {
  local pgid="$1"
  if [ -n "$pgid" ]; then
    echo "STOP_SERVER_GROUP pgid=$pgid"
    kill -TERM "-$pgid" >/dev/null 2>&1 || true
    sleep 8
    kill -KILL "-$pgid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  if [ -n "${CURRENT_SERVER_PGID:-}" ]; then
    stop_server_group "$CURRENT_SERVER_PGID"
    CURRENT_SERVER_PGID=""
  fi
}
trap cleanup EXIT

check_gpu5_free_or_die() {
  local used free pcount
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
  pcount="$(nvidia-smi pmon -c 1 | awk -v g="$GPU_ID" '$1==g && $2!="-" {print}' | wc -l)"
  echo "GPU5_CHECK used=${used}MiB free=${free}MiB real_process_count=${pcount}"
  if [ "$free" -lt 70000 ] || [ "$pcount" -ne 0 ]; then
    echo "GPU5_NOT_SAFE_ABORT"
    exit 1
  fi
}

port_free_or_die() {
  local port="$1"
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\])${port}$"; then
    echo "PORT_BUSY_ABORT port=$port"
    exit 1
  fi
}

find_snapshot_repo() {
  local repo="$1"
  local repo_dir="models--${repo//\//--}"
  find "$HOME/.cache/huggingface/hub/${repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true
}

wait_api() {
  local port="$1"
  local waited=0
  while true; do
    if curl -fs "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "VLLM_READY port=$port"
      return 0
    fi
    if [ "$waited" -ge 1200 ]; then
      echo "VLLM_TIMEOUT port=$port"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

start_server() {
  local model_path="$1"
  local label="$2"
  local port="$3"
  local max_model_len="$4"
  local gpu_util="$5"

  if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
    echo "SKIP_SERVER_MISSING label=$label path=$model_path"
    return 1
  fi

  check_gpu5_free_or_die
  port_free_or_die "$port"

  local log="logs/vllm_gpu5_final_gap_resume_${label}_${port}.log"

  echo
  echo "============================================================"
  echo "START_SERVER label=$label physical_gpu=$GPU_ID port=$port"
  echo "MODEL_PATH=$model_path"
  echo "LOG=$log"
  echo "============================================================"

  setsid bash -lc "
    export CUDA_VISIBLE_DEVICES='$GPU_ID'
    export HF_HOME='$HOME/.cache/huggingface'
    export HF_HUB_CACHE='$HOME/.cache/huggingface/hub'
    exec '$VLLM_PY' -m vllm.entrypoints.openai.api_server \
      --model '$model_path' \
      --served-model-name local-model \
      --host 127.0.0.1 \
      --port '$port' \
      --dtype auto \
      --max-model-len '$max_model_len' \
      --gpu-memory-utilization '$gpu_util' \
      --max-num-seqs 1 \
      --trust-remote-code
  " > "$log" 2>&1 &

  CURRENT_SERVER_PGID="$!"
  echo "$CURRENT_SERVER_PGID" > "$OUT/current_server_pgid.txt"

  if wait_api "$port"; then
    return 0
  else
    echo "SERVER_FAILED label=$label see=$log"
    stop_server_group "$CURRENT_SERVER_PGID"
    CURRENT_SERVER_PGID=""
    return 1
  fi
}

serve_and_run() {
  local model_path="$1"
  local label="$2"
  local port="$3"
  local max_model_len="${4:-4096}"
  local gpu_util="${5:-0.88}"

  if start_server "$model_path" "$label" "$port" "$max_model_len" "$gpu_util"; then
    "$CLIENT_PY" "$RUNPY" run-qa \
      --input "$PHASE2/stronger_model_eval_input.jsonl" \
      --output "$OUT/outputs/${label}_qa.jsonl" \
      --base-url "http://127.0.0.1:${port}/v1" \
      --served-model local-model \
      --model-label "$label" \
      --modes "direct,cot,hypmed_v4" \
      --max-per-dataset "$QA_MAX_PER_DATASET"

    "$CLIENT_PY" "$RUNPY" run-space \
      --input "$PHASE2/space_v4_stress_input.jsonl" \
      --output "$OUT/outputs/${label}_space_v4.jsonl" \
      --base-url "http://127.0.0.1:${port}/v1" \
      --served-model local-model \
      --model-label "$label" \
      --max-rows "$SPACE_MAX_ROWS" \
      --hybrid-skip-duplicates

    stop_server_group "$CURRENT_SERVER_PGID"
    CURRENT_SERVER_PGID=""
    sleep 15
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -i "$GPU_ID" || true
  else
    echo "SKIP_MODEL_RUN label=$label"
  fi
}

echo "FINAL_GAP_MEDICAL_RESUME_START $(date)"
echo "CUDA policy: only physical GPU $GPU_ID"

echo
echo "=== INITIAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
check_gpu5_free_or_die

"$CLIENT_PY" "$RUNPY" ensure-inputs

OPENBIOLLM="$(find_snapshot_repo 'aaditya/Llama3-OpenBioLLM-8B')"
MED42="$(find_snapshot_repo 'm42-health/Llama3-Med42-8B')"

echo
echo "=== MEDICAL MODEL SNAPSHOTS ==="
echo "OPENBIOLLM=$OPENBIOLLM"
echo "MED42=$MED42"

serve_and_run "$OPENBIOLLM" "openbiollm_llama3_8b" "$((PORT_BASE+1))" "4096" "0.88"
serve_and_run "$MED42" "med42_llama3_8b" "$((PORT_BASE+2))" "4096" "0.88"

"$CLIENT_PY" "$RUNPY" summarize

echo
echo "=== FINAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

echo "FINAL_GAP_MEDICAL_RESUME_OK"
echo "OUT=$OUT"
echo "SUMMARY=$OUT/FINAL_GAP_SUMMARY.md"
