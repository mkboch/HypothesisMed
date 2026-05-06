import json
import math
from pathlib import Path
from collections import Counter
import pandas as pd

def ci95(p, n):
    if n == 0:
        return ""
    se = math.sqrt(p * (1 - p) / n)
    return f"[{max(0,p-1.96*se):.3f}, {min(1,p+1.96*se):.3f}]"

def metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    def calc(sub):
        n = len(sub)
        valid_pred = [r for r in sub if r.get("pred_answer") is not None]
        acc = sum(r.get("pred_answer") == r.get("gold_answer") for r in sub) / n if n else 0
        parse_cov = len(valid_pred) / n if n else 0

        space_rows = [r for r in sub if r.get("pred_space_label") is not None]
        space_cov = len(space_rows) / n if n else 0
        space_acc = (
            sum(r.get("pred_space_label") == r.get("gold_space_label") for r in space_rows) / len(space_rows)
            if space_rows else None
        )

        high_wrong = [
            r for r in sub
            if r.get("pred_answer") is not None
            and r.get("pred_answer") != r.get("gold_answer")
            and float(r.get("confidence") or 0.0) >= 0.8
        ]
        wrong = [
            r for r in sub
            if r.get("pred_answer") is not None
            and r.get("pred_answer") != r.get("gold_answer")
        ]
        fcr_wrong_cond = len(high_wrong) / len(wrong) if wrong else 0

        return {
            "n": n,
            "answer_accuracy": round(acc, 4),
            "accuracy_ci95": ci95(acc, n),
            "parse_coverage": round(parse_cov, 4),
            "space_label_coverage": round(space_cov, 4),
            "space_label_accuracy": None if space_acc is None else round(space_acc, 4),
            "false_commitment_wrong_cond": round(fcr_wrong_cond, 4),
        }

    all_m = calc(rows)
    original = [r for r in rows if r.get("transform") == "original"]
    corrupt = [r for r in rows if r.get("transform") != "original"]

    return {
        "file": str(path),
        "model": rows[0].get("model") if rows else "",
        "method": rows[0].get("method") if rows else "",
        "all_variants": all_m,
        "original_only_standard_accuracy": calc(original),
        "corrupted_variants_reliability": calc(corrupt),
        "pred_answer_counts": dict(Counter(r.get("pred_answer") for r in rows)),
        "pred_space_counts": dict(Counter(r.get("pred_space_label") for r in rows)),
    }

files = sorted(Path("results/reparsed_paper_ready").glob("*medqa*.jsonl"))
summaries = [metrics(p) for p in files]

Path("results/paper_ready").mkdir(parents=True, exist_ok=True)

flat = []
for s in summaries:
    for section in ["all_variants", "original_only_standard_accuracy", "corrupted_variants_reliability"]:
        row = {
            "file": s["file"],
            "model": s["model"],
            "method": s["method"],
            "section": section,
        }
        row.update(s[section])
        flat.append(row)

df = pd.DataFrame(flat)
df.to_csv("results/paper_ready/medqa_paper_ready_summary.csv", index=False)

with open("results/paper_ready/medqa_paper_ready_summary.json", "w") as f:
    json.dump(summaries, f, indent=2)

print(df.to_string(index=False))
print("\nSaved:")
print("results/paper_ready/medqa_paper_ready_summary.csv")
print("results/paper_ready/medqa_paper_ready_summary.json")
