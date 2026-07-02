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
MGOUT="results/expanded_medgemma"
PHASE2="results/expanded_fix_vllm"

GPU_ID="5"
PORT_BASE="${PORT_BASE:-19550}"
QA_MAX_PER_DATASET="${QA_MAX_PER_DATASET:-1000}"
SPACE_MAX_ROWS="${SPACE_MAX_ROWS:-0}"

mkdir -p "$OUT/outputs" "$MGOUT" logs

source "$MGOUT/medgemma_model_paths.env"

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

wait_api() {
  local port="$1"
  local waited=0
  while true; do
    if curl -fs "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "VLLM_READY port=$port"
      return 0
    fi
    if [ "$waited" -ge 1800 ]; then
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
  local dtype="$6"

  if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
    echo "SKIP_SERVER_MISSING label=$label path=$model_path"
    return 1
  fi

  check_gpu5_free_or_die
  port_free_or_die "$port"

  local log="logs/vllm_gpu5_medgemma_${label}_${port}.log"

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
      --dtype '$dtype' \
      --max-model-len '$max_model_len' \
      --gpu-memory-utilization '$gpu_util' \
      --max-num-seqs 1 \
      --trust-remote-code \
      --disable-log-requests
  " > "$log" 2>&1 &

  CURRENT_SERVER_PGID="$!"
  echo "$CURRENT_SERVER_PGID" > "$MGOUT/current_medgemma_server_pgid.txt"

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
  local max_model_len="$4"
  local gpu_util="$5"
  local dtype="$6"

  if start_server "$model_path" "$label" "$port" "$max_model_len" "$gpu_util" "$dtype"; then
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

echo "MEDGEMMA_GPU5_RUN_START $(date)"
echo "CUDA policy: only physical GPU $GPU_ID"
echo "CLIENT_PY=$CLIENT_PY"
echo "VLLM_PY=$VLLM_PY"
echo "QA_MAX_PER_DATASET=$QA_MAX_PER_DATASET"
echo "SPACE_MAX_ROWS=$SPACE_MAX_ROWS"

echo
echo "=== INITIAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
check_gpu5_free_or_die

"$CLIENT_PY" "$RUNPY" ensure-inputs

echo
echo "=== MEDGEMMA PATHS ==="
echo "MEDGEMMA_1_5_4B_IT=${MEDGEMMA_1_5_4B_IT:-}"
echo "MEDGEMMA_27B_TEXT_IT=${MEDGEMMA_27B_TEXT_IT:-}"
echo "MEDGEMMA_4B_IT_LEGACY=${MEDGEMMA_4B_IT_LEGACY:-}"

serve_and_run "${MEDGEMMA_1_5_4B_IT:-}" "medgemma_1_5_4b_it" "$((PORT_BASE+1))" "4096" "0.84" "bfloat16"

serve_and_run "${MEDGEMMA_27B_TEXT_IT:-}" "medgemma_27b_text_it" "$((PORT_BASE+2))" "4096" "0.92" "bfloat16"

echo "SKIP legacy medgemma_4b_it because this account is not authorized for google/medgemma-4b-it."

"$CLIENT_PY" "$RUNPY" summarize

echo
echo "=== FINAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

echo "MEDGEMMA_GPU5_RUN_OK"
echo "OUT=$OUT"
echo "SUMMARY=$OUT/FINAL_GAP_SUMMARY.md"
