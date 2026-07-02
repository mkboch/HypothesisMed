#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/manikm/HypothesisMed

LOG="logs/claude_haiku_batch_run.log"
WATCHLOG="logs/claude_haiku_batch_watch.log"
STATE="results/expanded_claude_batch/claude_haiku_batch_state.json"

echo "CLAUDE_BATCH_WATCH_START $(date)" | tee -a "$WATCHLOG"

while true; do
  {
    echo
    echo "============================================================"
    echo "CHECK_TIME: $(date)"
    echo "============================================================"

    echo
    echo "=== TMUX SESSION STATUS ==="
    tmux ls 2>/dev/null | grep -E "hypmed_claude_haiku_batch|hypmed_claude_batch_watch" || true

    echo
    echo "=== LATEST BATCH STATUS FROM LOG ==="
    if [ -f "$LOG" ]; then
      grep -E "BATCH_SUBMITTED|BATCH_STATUS|RESULT_DOWNLOAD|CLAUDE_HAIKU_BATCH|Traceback|Error|error|FAILED|failed" "$LOG" | tail -20 || true
    else
      echo "LOG_NOT_FOUND: $LOG"
    fi

    echo
    echo "=== STATE FILE ==="
    python3 - <<'PY'
from pathlib import Path
import json

p = Path("results/expanded_claude_batch/claude_haiku_batch_state.json")
if not p.exists():
    print("NO_STATE_FILE")
else:
    try:
        s = json.loads(p.read_text())
        print("batch_id:", s.get("id"))
        print("processing_status:", s.get("processing_status"))
        print("request_counts:", s.get("request_counts"))
        print("created_at:", s.get("created_at"))
        print("ended_at:", s.get("ended_at"))
        print("expires_at:", s.get("expires_at"))
    except Exception as e:
        print("STATE_PARSE_ERROR:", repr(e))
PY

    echo
    echo "=== OUTPUT COUNTS IF AVAILABLE ==="
    python3 - <<'PY'
from pathlib import Path
for p in sorted(Path("results/expanded_claude_batch").glob("claude_haiku_4_5_batch*.jsonl")):
    try:
        n = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
        print(f"{p.name}: {n}")
    except Exception as e:
        print(f"{p.name}: ERROR {e}")
PY

  } | tee -a "$WATCHLOG"

  if [ -f "$LOG" ] && grep -q "CLAUDE_HAIKU_BATCH_RUN_OK" "$LOG"; then
    {
      echo
      echo "============================================================"
      echo "CLAUDE_BATCH_DONE_OK $(date)"
      echo "============================================================"

      echo
      echo "=== FINAL CLAUDE TABLES ==="
      /home/manikm/HypothesisMed/.venv/bin/python3 - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("results/expanded_claude_batch")
for name in [
    "table_claude_haiku_by_method.csv",
    "table_claude_haiku_fusion.csv",
    "table_claude_haiku_space_v4_overall.csv",
    "table_claude_haiku_space_v4_by_label.csv",
    "table_claude_haiku_actual_usage_cost.csv",
]:
    p = base / name
    print(f"\n--- {name} ---")
    if not p.exists():
        print("MISSING")
        continue
    print(pd.read_csv(p).to_string(index=False))
PY

      echo
      echo "WATCH CHECK OK"
    } | tee -a "$WATCHLOG"
    exit 0
  fi

  if [ -f "$LOG" ] && grep -E "Traceback|BUDGET_ABORT|AuthenticationError|Permission|insufficient|rate_limit|FAILED|failed" "$LOG" >/dev/null 2>&1; then
    echo "WATCH DETECTED POSSIBLE FAILURE. See $LOG" | tee -a "$WATCHLOG"
    exit 1
  fi

  sleep 300
done
