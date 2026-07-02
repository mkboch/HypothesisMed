#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/manikm/HypothesisMed")
OUTDIR = ROOT / "results" / "main_scaled"
OUTDIR.mkdir(parents=True, exist_ok=True)

files = [
    ("MedQA", "Direct", ROOT / "results/qwen2_5_7b_instruct_direct_medqa_main_large.jsonl"),
    ("MedQA", "CoT", ROOT / "results/qwen2_5_7b_instruct_cot_medqa_main_large.jsonl"),
    ("MedQA", "HypMed-v3", ROOT / "results/qwen2_5_7b_instruct_hypothesismed_v3_medqa_main_large.jsonl"),
    ("MedQA", "Fusion", ROOT / "results/fusion/qwen2_5_7b_instruct_fusion_majority_answer_hypmed_v3_space_medqa_main_large.jsonl"),

    ("MedMCQA", "Direct", ROOT / "results/qwen2_5_7b_instruct_direct_medmcqa_main_large.jsonl"),
    ("MedMCQA", "CoT", ROOT / "results/qwen2_5_7b_instruct_cot_medmcqa_main_large.jsonl"),
    ("MedMCQA", "HypMed-v3", ROOT / "results/qwen2_5_7b_instruct_hypothesismed_v3_medmcqa_main_large.jsonl"),
    ("MedMCQA", "Fusion", ROOT / "results/fusion/qwen2_5_7b_instruct_fusion_majority_answer_hypmed_v3_space_medmcqa_main_large.jsonl"),

    ("PubMedQA", "Direct", ROOT / "results/qwen2_5_7b_instruct_direct_pubmedqa_main_large.jsonl"),
    ("PubMedQA", "CoT", ROOT / "results/qwen2_5_7b_instruct_cot_pubmedqa_main_large.jsonl"),
    ("PubMedQA", "HypMed-v3", ROOT / "results/qwen2_5_7b_instruct_hypothesismed_v3_pubmedqa_main_large.jsonl"),
    ("PubMedQA", "Fusion", ROOT / "results/fusion/qwen2_5_7b_instruct_fusion_majority_answer_hypmed_v3_space_pubmedqa_main_large.jsonl"),
]

rows_out = []

for dataset, method, path in files:
    if not path.exists():
        print(f"MISSING: {path}")
        continue

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    n = len(rows)

    parsed = sum(1 for r in rows if r.get("pred_answer"))
    correct = sum(1 for r in rows if r.get("pred_answer") == r.get("gold_answer"))

    space_cov = sum(1 for r in rows if r.get("pred_space_label"))
    space_correct = sum(
        1 for r in rows
        if r.get("pred_space_label") and r.get("pred_space_label") == r.get("gold_space_label")
    )

    wrong = sum(
        1 for r in rows
        if r.get("pred_answer") and r.get("pred_answer") != r.get("gold_answer")
    )
    high_conf_wrong = sum(
        1 for r in rows
        if r.get("pred_answer")
        and r.get("pred_answer") != r.get("gold_answer")
        and float(r.get("confidence") or 0.0) >= 0.5
    )

    rows_out.append({
        "Model": "Qwen2.5-7B",
        "Dataset": dataset,
        "Method": method,
        "N": n,
        "Accuracy": round(correct / n, 4) if n else 0.0,
        "Parse coverage": round(parsed / n, 4) if n else 0.0,
        "SPACE coverage": round(space_cov / n, 4) if n else 0.0,
        "SPACE accuracy extracted": round(space_correct / space_cov, 4) if space_cov else 0.0,
        "False commitment wrong-cond": round(high_conf_wrong / wrong, 4) if wrong else 0.0,
    })

df = pd.DataFrame(rows_out)

csv_path = OUTDIR / "qwen_scaled_summary_by_dataset_method.csv"
tex_path = OUTDIR / "qwen_scaled_summary_by_dataset_method.tex"

df.to_csv(csv_path, index=False)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(df.to_latex(index=False, escape=True))

print("===== QWEN SCALED SUMMARY TABLE =====")
print(df.to_string(index=False))

print()
print(f"Saved CSV: {csv_path}")
print(f"Saved TeX: {tex_path}")

print()
print("===== WEIGHTED AGGREGATE BY METHOD =====")
agg = []
for method, g in df.groupby("Method"):
    total_n = g["N"].sum()
    agg.append({
        "Method": method,
        "N": int(total_n),
        "Weighted accuracy": round((g["Accuracy"] * g["N"]).sum() / total_n, 4),
        "Weighted parse coverage": round((g["Parse coverage"] * g["N"]).sum() / total_n, 4),
        "Weighted SPACE coverage": round((g["SPACE coverage"] * g["N"]).sum() / total_n, 4),
        "Weighted false commitment": round((g["False commitment wrong-cond"] * g["N"]).sum() / total_n, 4),
    })

agg_df = pd.DataFrame(agg).sort_values("Method")
agg_csv = OUTDIR / "qwen_scaled_weighted_aggregate_by_method.csv"
agg_tex = OUTDIR / "qwen_scaled_weighted_aggregate_by_method.tex"
agg_df.to_csv(agg_csv, index=False)
with open(agg_tex, "w", encoding="utf-8") as f:
    f.write(agg_df.to_latex(index=False, escape=True))

print(agg_df.to_string(index=False))
print()
print(f"Saved aggregate CSV: {agg_csv}")
print(f"Saved aggregate TeX: {agg_tex}")
