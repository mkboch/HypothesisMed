import json
import math
from pathlib import Path
from collections import Counter
import pandas as pd

OUTDIR = Path("results/final_multidataset")
OUTDIR.mkdir(parents=True, exist_ok=True)
Path("results/fusion").mkdir(parents=True, exist_ok=True)

def valid_letter(x):
    return isinstance(x, str) and x.strip().upper() in set("ABCDE")

def load_rows(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def result_file(model, method, dataset):
    return Path(f"results/{model}_{method}_{dataset}.jsonl")

def majority(vals):
    vals = [v.strip().upper() for v in vals if valid_letter(v)]
    if not vals:
        return None
    c = Counter(vals)
    top_n = max(c.values())
    top = [k for k, v in c.items() if v == top_n]
    for v in vals:
        if v in top:
            return v
    return top[0]

def create_fusion(model, dataset):
    direct = {r["id"]: r for r in load_rows(result_file(model, "direct", dataset))}
    cot = {r["id"]: r for r in load_rows(result_file(model, "cot", dataset))}
    v3_rows = load_rows(result_file(model, "hypothesismed_v3", dataset))
    v3 = {r["id"]: r for r in v3_rows}

    if not v3_rows:
        return None

    ids = [r["id"] for r in v3_rows]
    rows = []

    for i in ids:
        base = dict(v3[i])
        base["raw_output"] = ""
        base["parsed_output"] = {}
        ans = majority([
            direct.get(i, {}).get("pred_answer"),
            cot.get(i, {}).get("pred_answer"),
            v3.get(i, {}).get("pred_answer"),
        ])
        base["model"] = model
        base["method"] = "fusion_majority_answer_hypmed_v3_space"
        base["pred_answer"] = ans
        base["pred_space_label"] = v3.get(i, {}).get("pred_space_label")
        base["confidence"] = v3.get(i, {}).get("confidence", 0.0)
        rows.append(base)

    out = Path(f"results/fusion/{model}_fusion_majority_answer_hypmed_v3_space_{dataset}.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out

def ci95(p, n):
    if n == 0:
        return ""
    se = math.sqrt(p * (1 - p) / n)
    return f"[{max(0,p-1.96*se):.3f}, {min(1,p+1.96*se):.3f}]"

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
        "dataset": rows[0].get("dataset", Path(path).stem),
        "file": str(path),
        "model": rows[0].get("model"),
        "method": rows[0].get("method"),
        "n": n,
        "answer_accuracy": round(acc, 4),
        "accuracy_ci95": ci95(acc, n),
        "parse_coverage": round(parse_cov, 4),
        "space_label_coverage": round(space_cov, 4),
        "space_label_accuracy": "" if space_acc is None else round(space_acc, 4),
        "false_commitment_wrong_cond": round(fcr, 4),
    }

datasets = []
for p in sorted(Path("datasets/transformed").glob("*original1000.jsonl")):
    datasets.append(p.stem)

models = ["qwen2_5_7b_instruct", "deepseek_r1_qwen_32b"]
methods = ["direct", "cot", "hypothesismed_v3"]

fusion_paths = []
for model in models:
    for dataset in datasets:
        fp = create_fusion(model, dataset)
        if fp:
            fusion_paths.append(fp)

all_paths = []
for model in models:
    for dataset in datasets:
        for method in methods:
            p = result_file(model, method, dataset)
            if p.exists() and p.stat().st_size > 0:
                all_paths.append(p)

all_paths.extend(fusion_paths)

summary_rows = []
for p in all_paths:
    s = summarize(p)
    if s:
        summary_rows.append(s)

df = pd.DataFrame(summary_rows)
if len(df):
    df = df.sort_values(["dataset", "model", "answer_accuracy"], ascending=[True, True, False])

df.to_csv(OUTDIR / "all_results_by_dataset_model_method.csv", index=False)

# Aggregate across datasets for each model and method
agg_rows = []
if len(df):
    for (model, method), g in df.groupby(["model", "method"]):
        total_n = g["n"].sum()
        weighted_acc = (g["answer_accuracy"] * g["n"]).sum() / total_n
        weighted_parse = (g["parse_coverage"] * g["n"]).sum() / total_n
        weighted_space_cov = (g["space_label_coverage"] * g["n"]).sum() / total_n
        weighted_fcr = (g["false_commitment_wrong_cond"] * g["n"]).sum() / total_n
        agg_rows.append({
            "model": model,
            "method": method,
            "datasets": len(g),
            "total_n": int(total_n),
            "weighted_answer_accuracy": round(weighted_acc, 4),
            "weighted_parse_coverage": round(weighted_parse, 4),
            "weighted_space_label_coverage": round(weighted_space_cov, 4),
            "weighted_false_commitment_wrong_cond": round(weighted_fcr, 4),
        })

agg = pd.DataFrame(agg_rows)
if len(agg):
    agg = agg.sort_values(["model", "weighted_answer_accuracy"], ascending=[True, False])

agg.to_csv(OUTDIR / "aggregate_results_across_datasets.csv", index=False)

print("===== PER-DATASET RESULTS =====")
print(df.to_string(index=False) if len(df) else "No results found.")
print("\n===== AGGREGATE RESULTS =====")
print(agg.to_string(index=False) if len(agg) else "No aggregate results found.")
print("\nSaved:")
print(OUTDIR / "all_results_by_dataset_model_method.csv")
print(OUTDIR / "aggregate_results_across_datasets.csv")
