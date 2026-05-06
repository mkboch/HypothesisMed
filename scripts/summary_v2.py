import json, math
from pathlib import Path
import pandas as pd

def ci(p,n):
    if n == 0:
        return ""
    se = math.sqrt(p*(1-p)/n)
    return f"[{max(0,p-1.96*se):.3f}, {min(1,p+1.96*se):.3f}]"

def calc(rows):
    n = len(rows)
    if n == 0:
        return dict(n=0, answer_accuracy=0, accuracy_ci95="", parse_coverage=0, space_label_coverage=0, space_label_accuracy=None, false_commitment_wrong_cond=0)

    acc = sum(r.get("pred_answer") == r.get("gold_answer") for r in rows) / n
    parse_cov = sum(r.get("pred_answer") is not None for r in rows) / n

    space_rows = [r for r in rows if r.get("pred_space_label") is not None]
    space_cov = len(space_rows) / n
    space_acc = sum(r.get("pred_space_label") == r.get("gold_space_label") for r in space_rows) / len(space_rows) if space_rows else None

    wrong = [r for r in rows if r.get("pred_answer") is not None and r.get("pred_answer") != r.get("gold_answer")]
    high_wrong = [r for r in wrong if float(r.get("confidence") or 0) >= 0.8]
    fcr = len(high_wrong) / len(wrong) if wrong else 0

    return dict(
        n=n,
        answer_accuracy=round(acc,4),
        accuracy_ci95=ci(acc,n),
        parse_coverage=round(parse_cov,4),
        space_label_coverage=round(space_cov,4),
        space_label_accuracy=None if space_acc is None else round(space_acc,4),
        false_commitment_wrong_cond=round(fcr,4)
    )

rows_out = []
for p in sorted(Path("results/reparsed_v2").glob("*jsonl")) + sorted(Path("results/reparsed_paper_ready").glob("*jsonl")):
    if not p.exists() or p.stat().st_size == 0:
        continue
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if not rows:
        continue

    all_rows = rows
    orig = [r for r in rows if r.get("transform") == "original"]
    corrupt = [r for r in rows if r.get("transform") != "original"]

    for section, subset in [
        ("all_variants", all_rows),
        ("original_only_standard_accuracy", orig),
        ("corrupted_variants_reliability", corrupt),
    ]:
        m = calc(subset)
        rows_out.append({
            "file": str(p),
            "model": rows[0].get("model"),
            "method": rows[0].get("method"),
            "section": section,
            **m
        })

df = pd.DataFrame(rows_out)
Path("results/paper_ready_v2").mkdir(parents=True, exist_ok=True)
df.to_csv("results/paper_ready_v2/summary_v2.csv", index=False)

print(df.to_string(index=False))
print("\nSaved results/paper_ready_v2/summary_v2.csv")
