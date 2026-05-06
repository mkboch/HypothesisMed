import json
import math
from pathlib import Path
import pandas as pd

def ci95(p, n):
    if n == 0:
        return ""
    se = math.sqrt(p * (1 - p) / n)
    return f"[{max(0,p-1.96*se):.3f}, {min(1,p+1.96*se):.3f}]"

def calc(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    n = len(rows)
    acc = sum(r.get("pred_answer") == r.get("gold_answer") for r in rows) / n if n else 0
    parse_cov = sum(r.get("pred_answer") is not None for r in rows) / n if n else 0

    space_rows = [r for r in rows if r.get("pred_space_label") is not None]
    space_cov = len(space_rows) / n if n else 0
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
    fcr = len(high_wrong) / len(wrong) if wrong else 0

    return {
        "file": str(path),
        "model": rows[0].get("model") if rows else "",
        "method": rows[0].get("method") if rows else "",
        "n": n,
        "answer_accuracy": round(acc, 4),
        "accuracy_ci95": ci95(acc, n),
        "parse_coverage": round(parse_cov, 4),
        "space_label_coverage": round(space_cov, 4),
        "space_label_accuracy": None if space_acc is None else round(space_acc, 4),
        "false_commitment_wrong_cond": round(fcr, 4),
    }

files = sorted(Path("results").glob("*medqa_original1000*.jsonl"))
rows = [calc(f) for f in files if f.exists() and f.stat().st_size > 0]
df = pd.DataFrame(rows)

Path("results/paper_ready_clean").mkdir(parents=True, exist_ok=True)
df.to_csv("results/paper_ready_clean/medqa_original1000_summary.csv", index=False)

if len(df):
    print(df.sort_values("answer_accuracy", ascending=False).to_string(index=False))
else:
    print("No result files found.")

print("\nSaved results/paper_ready_clean/medqa_original1000_summary.csv")
