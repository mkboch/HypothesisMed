#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/manikm/HypothesisMed")
OUTDIR = ROOT / "results" / "main_scaled"
OUTDIR.mkdir(parents=True, exist_ok=True)

models = [
    ("Qwen2.5-7B", "qwen2_5_7b_instruct"),
    ("Phi-4-mini", "microsoft_phi_4_mini_instruct"),
]

datasets = [
    ("MedQA", "medqa_main_large"),
    ("MedMCQA", "medmcqa_main_large"),
    ("PubMedQA", "pubmedqa_main_large"),
]

methods = [
    ("Direct", "direct", "results/{model}_{method}_{stem}.jsonl"),
    ("CoT", "cot", "results/{model}_{method}_{stem}.jsonl"),
    ("HypMed-v3", "hypothesismed_v3", "results/{model}_{method}_{stem}.jsonl"),
    ("Fusion", "fusion_majority_answer_hypmed_v3_space", "results/fusion/{model}_fusion_majority_answer_hypmed_v3_space_{stem}.jsonl"),
]

def summarize_file(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
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

    return {
        "N": n,
        "Accuracy": correct / n if n else 0.0,
        "Parse coverage": parsed / n if n else 0.0,
        "SPACE coverage": space_cov / n if n else 0.0,
        "SPACE accuracy extracted": space_correct / space_cov if space_cov else 0.0,
        "False commitment wrong-cond": high_conf_wrong / wrong if wrong else 0.0,
    }

rows_out = []

for model_clean, model_key in models:
    for dataset_clean, stem in datasets:
        for method_clean, method_key, pattern in methods:
            rel = pattern.format(model=model_key, method=method_key, stem=stem)
            path = ROOT / rel
            if not path.exists():
                print(f"MISSING: {path}")
                continue

            s = summarize_file(path)
            rows_out.append({
                "Model": model_clean,
                "Dataset": dataset_clean,
                "Method": method_clean,
                "N": s["N"],
                "Accuracy": round(s["Accuracy"], 4),
                "Parse coverage": round(s["Parse coverage"], 4),
                "SPACE coverage": round(s["SPACE coverage"], 4),
                "SPACE accuracy extracted": round(s["SPACE accuracy extracted"], 4),
                "False commitment wrong-cond": round(s["False commitment wrong-cond"], 4),
            })

df = pd.DataFrame(rows_out)

csv_path = OUTDIR / "two_model_scaled_summary_by_dataset_method.csv"
tex_path = OUTDIR / "two_model_scaled_summary_by_dataset_method.tex"
df.to_csv(csv_path, index=False)
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(df.to_latex(index=False, escape=True))

print("===== TWO-MODEL SCALED SUMMARY TABLE =====")
print(df.to_string(index=False))

print()
print(f"Saved CSV: {csv_path}")
print(f"Saved TeX: {tex_path}")

agg_rows = []
for (model, method), g in df.groupby(["Model", "Method"]):
    total_n = g["N"].sum()
    agg_rows.append({
        "Model": model,
        "Method": method,
        "N": int(total_n),
        "Weighted accuracy": round((g["Accuracy"] * g["N"]).sum() / total_n, 4),
        "Weighted parse coverage": round((g["Parse coverage"] * g["N"]).sum() / total_n, 4),
        "Weighted SPACE coverage": round((g["SPACE coverage"] * g["N"]).sum() / total_n, 4),
        "Weighted SPACE accuracy extracted": round((g["SPACE accuracy extracted"] * g["N"]).sum() / total_n, 4),
        "Weighted false commitment": round((g["False commitment wrong-cond"] * g["N"]).sum() / total_n, 4),
    })

agg = pd.DataFrame(agg_rows).sort_values(["Model", "Method"])

agg_csv = OUTDIR / "two_model_scaled_weighted_aggregate_by_method.csv"
agg_tex = OUTDIR / "two_model_scaled_weighted_aggregate_by_method.tex"
agg.to_csv(agg_csv, index=False)
with open(agg_tex, "w", encoding="utf-8") as f:
    f.write(agg.to_latex(index=False, escape=True))

print()
print("===== TWO-MODEL WEIGHTED AGGREGATE BY METHOD =====")
print(agg.to_string(index=False))

print()
print(f"Saved aggregate CSV: {agg_csv}")
print(f"Saved aggregate TeX: {agg_tex}")

# Compact comparison: best answer-only baseline vs Fusion
compact = []
for model, g_model in agg.groupby("Model"):
    fusion = g_model[g_model["Method"] == "Fusion"].iloc[0]
    baselines = g_model[g_model["Method"].isin(["Direct", "CoT"])]
    best_base = baselines.sort_values("Weighted accuracy", ascending=False).iloc[0]
    compact.append({
        "Model": model,
        "Best Direct/CoT": best_base["Method"],
        "Best baseline acc.": best_base["Weighted accuracy"],
        "Fusion acc.": fusion["Weighted accuracy"],
        "Fusion minus baseline": round(fusion["Weighted accuracy"] - best_base["Weighted accuracy"], 4),
        "Fusion parse cov.": fusion["Weighted parse coverage"],
        "Fusion SPACE cov.": fusion["Weighted SPACE coverage"],
        "Fusion false commit.": fusion["Weighted false commitment"],
    })

compact_df = pd.DataFrame(compact)
compact_csv = OUTDIR / "two_model_scaled_compact_comparison.csv"
compact_tex = OUTDIR / "two_model_scaled_compact_comparison.tex"
compact_df.to_csv(compact_csv, index=False)
with open(compact_tex, "w", encoding="utf-8") as f:
    f.write(compact_df.to_latex(index=False, escape=True))

print()
print("===== COMPACT SCALED COMPARISON =====")
print(compact_df.to_string(index=False))

print()
print(f"Saved compact CSV: {compact_csv}")
print(f"Saved compact TeX: {compact_tex}")
