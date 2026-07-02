#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/manikm/HypothesisMed

CLIENT_PY="/home/manikm/HypothesisMed/.venv/bin/python3"
if [ -x "/home/manikm/miniconda3/envs/vllm311/bin/python" ]; then
  VLLM_PY="/home/manikm/miniconda3/envs/vllm311/bin/python"
else
  VLLM_PY="$CLIENT_PY"
fi

RUNPY="scripts/expanded_local_model_evaluation.py"
OUT="results/expanded_fix_vllm"
mkdir -p "$OUT/outputs" logs

GPU_ID="5"
PORT_BASE="${PORT_BASE:-19150}"
SPACE_MAX_ROWS="${SPACE_MAX_ROWS:-0}"
QA_MAX_PER_DATASET="${QA_MAX_PER_DATASET:-1000}"

check_gpu5_free_or_die() {
  local free used pcount
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
  pcount="$(nvidia-smi pmon -c 1 | awk -v g="$GPU_ID" '$1==g && $2!="-" {print}' | wc -l)"
  echo "GPU5_CHECK used=${used}MiB free=${free}MiB pmon_lines=${pcount}"
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

find_snapshot() {
  local repo_dir="$1"
  local p=""
  if [ -d "$HOME/.cache/huggingface/hub/${repo_dir}/snapshots" ]; then
    p="$(find "$HOME/.cache/huggingface/hub/${repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
  fi
  echo "$p"
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

stop_server_group() {
  local pgid="$1"
  if [ -n "$pgid" ]; then
    kill -TERM "-$pgid" >/dev/null 2>&1 || true
    sleep 8
    kill -KILL "-$pgid" >/dev/null 2>&1 || true
    wait "$pgid" >/dev/null 2>&1 || true
  fi
}

serve_and_run() {
  local model_path="$1"
  local label="$2"
  local task="$3"
  local port="$4"
  local max_model_len="$5"
  local gpu_util="$6"

  if [ ! -d "$model_path" ]; then
    echo "SKIP_MISSING_MODEL label=$label path=$model_path"
    return 0
  fi

  check_gpu5_free_or_die
  port_free_or_die "$port"

  local log="logs/vllm_gpu5_${label}_${task}_${port}.log"
  echo
  echo "============================================================"
  echo "START_SERVER label=$label task=$task physical_gpu=$GPU_ID port=$port"
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
      --trust-remote-code
  " > "$log" 2>&1 &

  local server_pgid="$!"

  if wait_api "$port"; then
    if [ "$task" = "space" ]; then
      "$CLIENT_PY" "$RUNPY" run-space \
        --input "$OUT/space_v4_stress_input.jsonl" \
        --output "$OUT/outputs/${label}_space_v4.jsonl" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --served-model local-model \
        --model-label "$label" \
        --max-rows "$SPACE_MAX_ROWS" \
        --hybrid-skip-duplicates
    elif [ "$task" = "qa" ]; then
      "$CLIENT_PY" "$RUNPY" run-qa \
        --input "$OUT/stronger_model_eval_input.jsonl" \
        --output "$OUT/outputs/${label}_qa.jsonl" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --served-model local-model \
        --model-label "$label" \
        --modes "direct,cot,hypmed_v4" \
        --max-per-dataset "$QA_MAX_PER_DATASET"
    else
      echo "UNKNOWN_TASK $task"
    fi
  else
    echo "SERVER_FAILED label=$label task=$task see=$log"
  fi

  echo "STOP_SERVER label=$label task=$task pgid=$server_pgid"
  stop_server_group "$server_pgid"

  sleep 15
  echo "GPU5_AFTER_STOP"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -i "$GPU_ID" || true
}

echo "PHASE2_GPU5_ALL_REMAINING_START $(date)"
echo "CLIENT_PY=$CLIENT_PY"
echo "VLLM_PY=$VLLM_PY"
echo "CUDA policy: only physical GPU $GPU_ID will be used by this script."

echo
echo "=== INITIAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

check_gpu5_free_or_die

"$CLIENT_PY" "$RUNPY" build-inputs

QWEN25="$(find_snapshot 'models--Qwen--Qwen2.5-7B-Instruct')"
if [ -z "$QWEN25" ]; then
  QWEN25="$(find_snapshot 'models--qwen--Qwen2.5-7B-Instruct')"
fi
PHI4="$(find_snapshot 'models--microsoft--Phi-4-mini-instruct')"
QWEN314="$(find_snapshot 'models--Qwen--Qwen3-14B')"
QWEN330="$(find_snapshot 'models--Qwen--Qwen3-30B-A3B')"
QWEN36="$(find_snapshot 'models--Qwen--Qwen3.6-35B-A3B')"

echo
echo "=== MODEL SNAPSHOTS ==="
echo "QWEN25=$QWEN25"
echo "PHI4=$PHI4"
echo "QWEN314=$QWEN314"
echo "QWEN330=$QWEN330"
echo "QWEN36=$QWEN36"

serve_and_run "$QWEN25" "qwen2_5_7b_instruct" "space" "$((PORT_BASE+1))" "4096" "0.80"
serve_and_run "$PHI4" "phi_4_mini_instruct" "space" "$((PORT_BASE+2))" "4096" "0.80"

serve_and_run "$QWEN314" "qwen3_14b" "qa" "$((PORT_BASE+3))" "4096" "0.88"

if [ -n "$QWEN330" ]; then
  serve_and_run "$QWEN330" "qwen3_30b_a3b" "qa" "$((PORT_BASE+4))" "3072" "0.92" || true
else
  echo "SKIP qwen3_30b_a3b missing"
fi

if [ -n "$QWEN36" ]; then
  serve_and_run "$QWEN36" "qwen3_6_35b_a3b" "qa" "$((PORT_BASE+5))" "2048" "0.95" || true
else
  echo "SKIP qwen3_6_35b_a3b missing"
fi

"$CLIENT_PY" "$RUNPY" summarize

echo
echo "=== FINAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

echo "PHASE2_GPU5_ALL_REMAINING_OK"
echo "OUT=$OUT"
echo "SUMMARY=$OUT/PHASE2_SUMMARY.md"
