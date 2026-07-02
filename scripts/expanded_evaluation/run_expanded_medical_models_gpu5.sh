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
mkdir -p "$OUT/outputs" logs

GPU_ID="5"
PORT_BASE="${PORT_BASE:-19250}"
QA_MAX_PER_DATASET="${QA_MAX_PER_DATASET:-1000}"
SPACE_MAX_ROWS="${SPACE_MAX_ROWS:-0}"
SC_MAX_PER_DATASET="${SC_MAX_PER_DATASET:-300}"
SC_K="${SC_K:-5}"

echo "FINAL_GAP_START $(date)"
echo "CUDA policy: only physical GPU ${GPU_ID}"
echo "CLIENT_PY=$CLIENT_PY"
echo "VLLM_PY=$VLLM_PY"

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

find_python_with_hf() {
  for p in "$CLIENT_PY" "$VLLM_PY" python3; do
    if "$p" - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
    then
      echo "$p"
      return 0
    fi
  done
  echo ""
}

download_or_skip() {
  local repo="$1"
  local label="$2"
  local snap
  snap="$(find_snapshot_repo "$repo")"
  if [ -n "$snap" ] && [ -d "$snap" ]; then
    echo "$snap"
    return 0
  fi

  local DL_PY
  DL_PY="$(find_python_with_hf)"
  if [ -z "$DL_PY" ]; then
    echo "DOWNLOAD_SKIP_NO_HUGGINGFACE_HUB label=$label repo=$repo" >&2
    echo ""
    return 0
  fi

  echo "DOWNLOAD_START label=$label repo=$repo using=$DL_PY" >&2
  if timeout 7200 "$DL_PY" - <<PY >&2
from huggingface_hub import snapshot_download
repo = "$repo"
path = snapshot_download(repo_id=repo, resume_download=True, local_files_only=False)
print(path)
PY
  then
    snap="$(find_snapshot_repo "$repo")"
    if [ -n "$snap" ] && [ -d "$snap" ]; then
      echo "DOWNLOAD_OK label=$label path=$snap" >&2
      echo "$snap"
      return 0
    fi
  fi

  echo "DOWNLOAD_FAILED_OR_GATED label=$label repo=$repo" >&2
  echo ""
  return 0
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

  local log="logs/vllm_gpu5_final_gap_${label}_${port}.log"
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
      --trust-remote-code
  " > "$log" 2>&1 &

  local pgid="$!"
  echo "$pgid" > "$OUT/current_server_pgid.txt"

  if wait_api "$port"; then
    echo "$pgid"
    return 0
  else
    echo "SERVER_FAILED label=$label see=$log"
    stop_server_group "$pgid"
    return 1
  fi
}

run_medical_model() {
  local model_path="$1"
  local label="$2"
  local port="$3"
  local max_model_len="${4:-4096}"
  local gpu_util="${5:-0.88}"

  local pgid
  if ! pgid="$(start_server "$model_path" "$label" "$port" "$max_model_len" "$gpu_util")"; then
    echo "SKIP_MODEL_RUN label=$label"
    return 0
  fi

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

  echo "STOP_SERVER label=$label pgid=$pgid"
  stop_server_group "$pgid"
  sleep 15
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -i "$GPU_ID" || true
}

run_qwen3_14b_extra() {
  local model_path="$1"
  local label="qwen3_14b_extra"
  local port="$2"

  local pgid
  if ! pgid="$(start_server "$model_path" "$label" "$port" "4096" "0.88")"; then
    echo "SKIP_QWEN3_14B_EXTRA"
    return 0
  fi

  "$CLIENT_PY" "$RUNPY" run-space \
    --input "$PHASE2/space_v4_stress_input.jsonl" \
    --output "$OUT/outputs/qwen3_14b_space_v4.jsonl" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --served-model local-model \
    --model-label "qwen3_14b" \
    --max-rows "$SPACE_MAX_ROWS" \
    --hybrid-skip-duplicates

  "$CLIENT_PY" "$RUNPY" run-self-consistency \
    --input "$PHASE2/stronger_model_eval_input.jsonl" \
    --output "$OUT/outputs/qwen3_14b_self_consistency_k${SC_K}.jsonl" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --served-model local-model \
    --model-label "qwen3_14b" \
    --max-per-dataset "$SC_MAX_PER_DATASET" \
    --k "$SC_K" \
    --temperature 0.7 \
    --top-p 0.95

  echo "STOP_SERVER label=$label pgid=$pgid"
  stop_server_group "$pgid"
  sleep 15
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -i "$GPU_ID" || true
}

echo
echo "=== INITIAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
check_gpu5_free_or_die

"$CLIENT_PY" "$RUNPY" ensure-inputs

QWEN314="$(find_snapshot_repo 'Qwen/Qwen3-14B')"
OPENBIOLLM="$(download_or_skip 'aaditya/Llama3-OpenBioLLM-8B' 'openbiollm_llama3_8b')"
MED42="$(download_or_skip 'm42-health/Llama3-Med42-8B' 'med42_llama3_8b')"
MEDGEMMA="$(download_or_skip 'google/medgemma-4b-it' 'medgemma_4b_it')"

echo
echo "=== MODEL SNAPSHOTS ==="
echo "QWEN314=$QWEN314"
echo "OPENBIOLLM=$OPENBIOLLM"
echo "MED42=$MED42"
echo "MEDGEMMA=$MEDGEMMA"

run_qwen3_14b_extra "$QWEN314" "$((PORT_BASE+1))"

if [ -n "$OPENBIOLLM" ]; then
  run_medical_model "$OPENBIOLLM" "openbiollm_llama3_8b" "$((PORT_BASE+2))" "4096" "0.88"
else
  echo "SKIP openbiollm_llama3_8b unavailable"
fi

if [ -n "$MED42" ]; then
  run_medical_model "$MED42" "med42_llama3_8b" "$((PORT_BASE+3))" "4096" "0.88"
else
  echo "SKIP med42_llama3_8b unavailable"
fi

if [ -n "$MEDGEMMA" ]; then
  run_medical_model "$MEDGEMMA" "medgemma_4b_it" "$((PORT_BASE+4))" "4096" "0.82" || true
else
  echo "SKIP medgemma_4b_it unavailable"
fi

"$CLIENT_PY" "$RUNPY" summarize

echo
echo "=== FINAL GPU STATUS ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

echo "FINAL_GAP_OK"
echo "OUT=$OUT"
echo "SUMMARY=$OUT/FINAL_GAP_SUMMARY.md"
