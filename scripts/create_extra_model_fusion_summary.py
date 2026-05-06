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
