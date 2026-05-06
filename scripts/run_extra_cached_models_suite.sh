#!/usr/bin/env bash
set -u

cd ~/HypothesisMed
source /home/manikm/HypothesisMed/.venv/bin/activate
export PYTHONPATH=.

echo "===== 1. Search cached HF models ====="

cat > scripts/inventory_cached_text_models.py <<'PY'
import os
import json
from pathlib import Path

roots = []
for env in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"]:
    v = os.environ.get(env)
    if v:
        roots.append(Path(v))

roots += [
    Path.home() / ".cache/huggingface/hub",
    Path("/home/manikm/lab_chatbot_h100/cache/hf/hub"),
    Path("/scratch/manikm/cache/hf/hub"),
    Path("/scratch/manikm/hf/hub"),
    Path("/scratch/manikm/.cache/huggingface/hub"),
]

seen = set()
models = []

bad_terms = [
    "vl", "vision", "ocr", "clip", "siglip", "whisper", "wav", "audio",
    "embed", "embedding", "reranker", "bge", "e5", "bert", "roberta",
    "deberta", "layout", "sam", "diffusion", "sdxl"
]

priority_terms = [
    "qwen3",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "gemma-2-9b",
    "llama-3.1-8b",
    "llama-3-8b",
    "mistral-7b",
    "mixtral",
]

exclude_terms = [
    "qwen2.5-7b-instruct",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-ocr",
    "qwen3-vl",
]

for root in roots:
    if not root.exists():
        continue

    for p in root.glob("models--*"):
        name = p.name.replace("models--", "").replace("--", "/")
        lname = name.lower()

        if name in seen:
            continue
        seen.add(name)

        if any(x in lname for x in bad_terms):
            continue
        if any(x in lname for x in exclude_terms):
            continue

        snapshots = p / "snapshots"
        if not snapshots.exists():
            continue

        snap_dirs = [s for s in snapshots.iterdir() if s.is_dir()]
        if not snap_dirs:
            continue

        # choose newest snapshot by modification time
        snap = max(snap_dirs, key=lambda x: x.stat().st_mtime)

        has_config = (snap / "config.json").exists()
        has_tokenizer = (
            (snap / "tokenizer.json").exists()
            or (snap / "tokenizer.model").exists()
            or (snap / "vocab.json").exists()
        )
        has_weights = bool(list(snap.glob("*.safetensors"))) or bool(list(snap.glob("*.bin")))

        if not (has_config and has_tokenizer and has_weights):
            continue

        score = 0
        for i, term in enumerate(priority_terms):
            if term in lname:
                score += 100 - i

        # prefer instruct/chat models
        if "instruct" in lname or "it" in lname or "chat" in lname:
            score += 20

        models.append({
            "name": name,
            "local_path": str(snap),
            "cache_root": str(root),
            "score": score,
            "mtime": snap.stat().st_mtime
        })

models = sorted(models, key=lambda x: (x["score"], x["mtime"]), reverse=True)

Path("results/local_extra_models").mkdir(parents=True, exist_ok=True)
with open("results/local_extra_models/cached_text_model_inventory.json", "w") as f:
    json.dump(models, f, indent=2)

selected = models[:3]
with open("results/local_extra_models/selected_extra_models.json", "w") as f:
    json.dump(selected, f, indent=2)

print("===== Cached text model candidates =====")
for m in models[:20]:
    print(f'{m["score"]:4d}  {m["name"]}  {m["local_path"]}')

print("\n===== Selected extra models =====")
for m in selected:
    print(f'{m["name"]}  {m["local_path"]}')
PY

python scripts/inventory_cached_text_models.py

echo
echo "===== 2. Create local vLLM evaluator for arbitrary cached models ====="

cat > scripts/run_local_cached_model_experiment.py <<'PY'
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.methods.prompts import build_prompt
from src.evaluation.parser import parse_output

def safe_key(name):
    name = name.lower()
    name = name.replace("/", "_").replace("-", "_").replace(".", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--method", required=True, choices=["direct", "cot", "hypothesismed_v3"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    model_key = safe_key(args.model_name)
    dataset_stem = Path(args.data).stem
    out_path = Path("results") / f"{model_key}_{args.method}_{dataset_stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and sum(1 for _ in out_path.open()) >= args.max_samples:
        print(f"[SKIP] Existing complete file: {out_path}")
        return

    rows = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.max_samples:
                break

    print(f"[INFO] model_name={args.model_name}")
    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] method={args.method}")
    print(f"[INFO] data={args.data}")
    print(f"[INFO] output={out_path}")
    print(f"[INFO] n={len(rows)}")

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        disable_log_stats=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    with out_path.open("w") as out:
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i+args.batch_size]
            prompts = [build_prompt(args.method, r) for r in batch]
            outputs = llm.generate(prompts, sampling)

            for r, o in zip(batch, outputs):
                raw = o.outputs[0].text if o.outputs else ""
                parsed = parse_output(raw)

                result = dict(r)
                result["model"] = model_key
                result["model_name"] = args.model_name
                result["model_path"] = args.model_path
                result["method"] = args.method
                result["raw_output"] = raw
                result["parsed_output"] = parsed
                result["pred_answer"] = parsed.get("answer")
                result["pred_space_label"] = parsed.get("space_label")
                result["confidence"] = parsed.get("confidence", 0.0)

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
PY

echo
echo "===== 3. Create extra-model fusion and summary script ====="

cat > scripts/create_extra_model_fusion_summary.py <<'PY'
import json
import math
from pathlib import Path
from collections import Counter
import pandas as pd

def valid_letter(x):
    return isinstance(x, str) and x.strip().upper() in set("ABCDE")

def load_rows(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def majority(vals):
    vals = [v.strip().upper() for v in vals if valid_letter(v)]
    if not vals:
        return None
    c = Counter(vals)
    top_count = max(c.values())
    top = [k for k, v in c.items() if v == top_count]
    for v in vals:
        if v in top:
            return v
    return top[0]

def ci95(p, n):
    if n == 0:
        return ""
    se = math.sqrt(p * (1 - p) / n)
    return f"[{max(0, p - 1.96*se):.3f}, {min(1, p + 1.96*se):.3f}]"

def summarize(path):
    rows = load_rows(path)
    if not rows:
        return None

    n = len(rows)
    acc = sum(r.get("pred_answer") == r.get("gold_answer") for r in rows) / n
    parse_cov = sum(r.get("pred_answer") is not None for r in rows) / n

    space_rows = [r for r in rows if r.get("pred_space_label") is not None]
    space_cov = len(space_rows) / n
    space_acc = (
        sum(r.get("pred_space_label") == r.get("gold_space_label") for r in space_rows) / len(space_rows)
        if space_rows else None
    )

    wrong = [
        r for r in rows
        if r.get("pred_answer") is not None and r.get("pred_answer") != r.get("gold_answer")
    ]
    high_wrong = [
        r for r in wrong
        if float(r.get("confidence") or 0.0) >= 0.8
    ]
    fcr = len(high_wrong) / len(wrong) if wrong else 0.0

    return {
        "dataset": rows[0].get("dataset"),
        "file": str(path),
        "model": rows[0].get("model"),
        "model_name": rows[0].get("model_name", rows[0].get("model")),
        "method": rows[0].get("method"),
        "n": n,
        "answer_accuracy": round(acc, 4),
        "accuracy_ci95": ci95(acc, n),
        "parse_coverage": round(parse_cov, 4),
        "space_label_coverage": round(space_cov, 4),
        "space_label_accuracy": "" if space_acc is None else round(space_acc, 4),
        "false_commitment_wrong_cond": round(fcr, 4),
    }

def create_fusion_for_model_dataset(model_key, dataset_stem):
    direct_path = Path(f"results/{model_key}_direct_{dataset_stem}.jsonl")
    cot_path = Path(f"results/{model_key}_cot_{dataset_stem}.jsonl")
    v3_path = Path(f"results/{model_key}_hypothesismed_v3_{dataset_stem}.jsonl")

    direct_rows = load_rows(direct_path)
    cot_rows = load_rows(cot_path)
    v3_rows = load_rows(v3_path)

    if not v3_rows:
        return None

    direct = {r["id"]: r for r in direct_rows}
    cot = {r["id"]: r for r in cot_rows}
    v3 = {r["id"]: r for r in v3_rows}

    rows = []
    for r0 in v3_rows:
        i = r0["id"]
        r = dict(r0)
        r["raw_output"] = ""
        r["parsed_output"] = {}

        ans = majority([
            direct.get(i, {}).get("pred_answer"),
            cot.get(i, {}).get("pred_answer"),
            v3.get(i, {}).get("pred_answer"),
        ])

        r["method"] = "fusion_majority_answer_hypmed_v3_space"
        r["pred_answer"] = ans
        r["pred_space_label"] = v3.get(i, {}).get("pred_space_label")
        r["confidence"] = v3.get(i, {}).get("confidence", 0.0)
        rows.append(r)

    Path("results/fusion").mkdir(parents=True, exist_ok=True)
    out = Path(f"results/fusion/{model_key}_fusion_majority_answer_hypmed_v3_space_{dataset_stem}.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out

# Detect all model keys from extra local result files
paths = sorted(Path("results").glob("*_hypothesismed_v3_*original1000.jsonl"))
model_dataset_pairs = []

for p in paths:
    name = p.name
    if name.startswith("qwen2_5_7b_instruct_") or name.startswith("deepseek_r1_qwen_32b_"):
        continue

    parts = name.replace(".jsonl", "")
    marker = "_hypothesismed_v3_"
    if marker not in parts:
        continue

    model_key, dataset_stem = parts.split(marker, 1)
    model_dataset_pairs.append((model_key, dataset_stem))

fusion_paths = []
for model_key, dataset_stem in sorted(set(model_dataset_pairs)):
    fp = create_fusion_for_model_dataset(model_key, dataset_stem)
    if fp:
        fusion_paths.append(fp)

# Include previous final results plus new extra model results
all_paths = []
for p in Path("results").glob("*original1000.jsonl"):
    if "smoke" not in p.name:
        all_paths.append(p)
for p in Path("results/fusion").glob("*original1000.jsonl"):
    if "smoke" not in p.name:
        all_paths.append(p)

summary_rows = []
seen = set()
for p in sorted(all_paths):
    if str(p) in seen:
        continue
    seen.add(str(p))
    s = summarize(p)
    if s:
        summary_rows.append(s)

df = pd.DataFrame(summary_rows)
if len(df):
    df = df.sort_values(["model", "dataset", "answer_accuracy"], ascending=[True, True, False])

Path("results/final_multimodel_extra").mkdir(parents=True, exist_ok=True)
df.to_csv("results/final_multimodel_extra/all_models_all_datasets_methods.csv", index=False)

agg_rows = []
if len(df):
    for (model, model_name, method), g in df.groupby(["model", "model_name", "method"]):
        total_n = g["n"].sum()
        agg_rows.append({
            "model": model,
            "model_name": model_name,
            "method": method,
            "datasets": int(g["dataset"].nunique()),
            "total_n": int(total_n),
            "weighted_answer_accuracy": round((g["answer_accuracy"] * g["n"]).sum() / total_n, 4),
            "weighted_parse_coverage": round((g["parse_coverage"] * g["n"]).sum() / total_n, 4),
            "weighted_space_label_coverage": round((g["space_label_coverage"] * g["n"]).sum() / total_n, 4),
            "weighted_false_commitment_wrong_cond": round((g["false_commitment_wrong_cond"] * g["n"]).sum() / total_n, 4),
        })

agg = pd.DataFrame(agg_rows)
if len(agg):
    agg = agg.sort_values(["method", "weighted_answer_accuracy"], ascending=[True, False])

agg.to_csv("results/final_multimodel_extra/aggregate_all_models.csv", index=False)

print("===== AGGREGATE ALL MODELS =====")
print(agg.to_string(index=False) if len(agg) else "No aggregate rows.")
print("\n===== PER-DATASET ALL MODELS =====")
print(df.to_string(index=False) if len(df) else "No per-dataset rows.")
PY

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '$2+0 > 60000 {gsub(/ /,"",$1); print $1; exit}'
}

echo
echo "===== 4. Run selected extra cached models ====="

python - <<'PY' > /tmp/extra_models.tsv
import json
from pathlib import Path

p = Path("results/local_extra_models/selected_extra_models.json")
models = json.loads(p.read_text()) if p.exists() else []

for m in models:
    print(m["name"] + "\t" + m["local_path"])
PY

cat /tmp/extra_models.tsv

if [ ! -s /tmp/extra_models.tsv ]; then
  echo "No extra cached text models found. Stop."
  exit 0
fi

DATASETS=("medqa_original1000" "medmcqa_original1000" "pubmedqa_original1000")
METHODS=("direct" "cot" "hypothesismed_v3")

while IFS=$'\t' read -r MODEL_NAME MODEL_PATH; do
  MODEL_KEY=$(python - <<PY
import re
name = """$MODEL_NAME""".lower().replace("/", "_").replace("-", "_").replace(".", "_")
name = re.sub(r"[^a-z0-9_]+", "_", name)
name = re.sub(r"_+", "_", name).strip("_")
print(name)
PY
)

  echo
  echo "============================================================"
  echo "MODEL: $MODEL_NAME"
  echo "KEY:   $MODEL_KEY"
  echo "PATH:  $MODEL_PATH"
  echo "============================================================"

  for DATASET in "${DATASETS[@]}"; do
    DATA_PATH="datasets/transformed/${DATASET}.jsonl"
    [ -s "$DATA_PATH" ] || { echo "Missing $DATA_PATH, skip"; continue; }

    for METHOD in "${METHODS[@]}"; do
      OUT="results/${MODEL_KEY}_${METHOD}_${DATASET}.jsonl"
      if [ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 1000 ]; then
        echo "[SKIP] $OUT already complete"
        continue
      fi

      FREE_GPU=$(pick_gpu)
      while [ -z "$FREE_GPU" ]; do
        echo "No GPU with >60GB free. Sleeping 120 sec..."
        sleep 120
        FREE_GPU=$(pick_gpu)
      done

      export CUDA_VISIBLE_DEVICES=$FREE_GPU
      echo "RUN model=$MODEL_NAME method=$METHOD dataset=$DATASET gpu=$FREE_GPU"

      MAXTOK=512
      if [ "$METHOD" = "direct" ]; then MAXTOK=256; fi
      if [ "$METHOD" = "hypothesismed_v3" ]; then MAXTOK=256; fi

      python scripts/run_local_cached_model_experiment.py \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --method "$METHOD" \
        --data "$DATA_PATH" \
        --max_samples 1000 \
        --batch_size 1 \
        --max_tokens "$MAXTOK" || true

      python scripts/reparse_results_inplace.py --glob "$OUT" || true
      python scripts/create_extra_model_fusion_summary.py || true
    done
  done

done < /tmp/extra_models.tsv

echo
echo "===== 5. Final all-model summary ====="
python scripts/create_extra_model_fusion_summary.py

echo
echo "===== FINAL AGGREGATE ALL MODELS ====="
cat results/final_multimodel_extra/aggregate_all_models.csv

echo
echo "===== FINAL PER-DATASET ALL MODELS ====="
cat results/final_multimodel_extra/all_models_all_datasets_methods.csv
