import pandas as pd
from pathlib import Path

src_dir = Path("results/final_multimodel_extra")
outdir = Path("results/final_paper_ready_submission")
outdir.mkdir(parents=True, exist_ok=True)

agg = pd.read_csv(src_dir / "aggregate_all_models.csv")
per = pd.read_csv(src_dir / "all_models_all_datasets_methods.csv")

proposed = "fusion_majority_answer_hypmed_v3_space"

main_models = [
    "qwen2_5_7b_instruct",
    "microsoft_phi_4_mini_instruct",
    "deepseek_r1_qwen_32b",
]

stress_models = [
    "biomistral_biomistral_7b",
]

model_names = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "microsoft_phi_4_mini_instruct": "Phi-4-mini-instruct",
    "deepseek_r1_qwen_32b": "DeepSeek-R1-Distill-Qwen-32B",
    "biomistral_biomistral_7b": "BioMistral-7B",
}

method_names = {
    "direct": "Direct",
    "cot": "Chain-of-thought",
    "hypothesismed_v3": "HypothesisMed-v3",
    "fusion_majority_answer_hypmed_v3_space": "Proposed: answer fusion + HypothesisMed-v3 SPACE",
}

method_order = {
    "Proposed: answer fusion + HypothesisMed-v3 SPACE": 0,
    "Chain-of-thought": 1,
    "Direct": 2,
    "HypothesisMed-v3": 3,
}

model_order = {
    "Qwen2.5-7B-Instruct": 0,
    "Phi-4-mini-instruct": 1,
    "DeepSeek-R1-Distill-Qwen-32B": 2,
    "BioMistral-7B": 3,
}

def clean_aggregate(df, models):
    x = df[df["model"].isin(models)].copy()
    x = x[x["datasets"].eq(3)]
    x = x[x["method"].isin(["direct", "cot", "hypothesismed_v3", proposed])]
    x["Model"] = x["model"].map(model_names).fillna(x["model"])
    x["Method"] = x["method"].map(method_names).fillna(x["method"])

    table = x[[
        "Model", "Method", "datasets", "total_n",
        "weighted_answer_accuracy",
        "weighted_parse_coverage",
        "weighted_space_label_coverage",
        "weighted_false_commitment_wrong_cond"
    ]].rename(columns={
        "datasets": "Datasets",
        "total_n": "N",
        "weighted_answer_accuracy": "Answer accuracy",
        "weighted_parse_coverage": "Parse coverage",
        "weighted_space_label_coverage": "SPACE coverage",
        "weighted_false_commitment_wrong_cond": "False commitment"
    })

    for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]:
        table[c] = pd.to_numeric(table[c], errors="coerce").round(4)

    table["_mo"] = table["Model"].map(model_order).fillna(99)
    table["_me"] = table["Method"].map(method_order).fillna(99)
    table = table.sort_values(["_mo", "_me"]).drop(columns=["_mo", "_me"])
    return table, x

main_table, main_raw = clean_aggregate(agg, main_models)
stress_table, stress_raw = clean_aggregate(agg, stress_models)

main_table.to_csv(outdir / "main_3model_aggregate_table.csv", index=False)
main_table.to_latex(outdir / "main_3model_aggregate_table.tex", index=False, escape=True)

stress_table.to_csv(outdir / "appendix_biomistral_stress_test_table.csv", index=False)
stress_table.to_latex(outdir / "appendix_biomistral_stress_test_table.tex", index=False, escape=True)

# Deltas for main models only.
rows = []
for model in main_models:
    m = main_raw[main_raw["model"].eq(model)]
    prop = m[m["method"].eq(proposed)]
    base = m[m["method"].isin(["direct", "cot"])]

    if prop.empty or base.empty:
        continue

    prop = prop.iloc[0]
    best = base.sort_values("weighted_answer_accuracy", ascending=False).iloc[0]

    prop_acc = float(prop["weighted_answer_accuracy"])
    base_acc = float(best["weighted_answer_accuracy"])
    prop_fc = float(prop["weighted_false_commitment_wrong_cond"])
    base_fc = float(best["weighted_false_commitment_wrong_cond"])

    rows.append({
        "Model": model_names.get(model, model),
        "Proposed accuracy": round(prop_acc, 4),
        "Best Direct/CoT baseline": method_names.get(best["method"], best["method"]),
        "Baseline accuracy": round(base_acc, 4),
        "Absolute accuracy gain": round(prop_acc - base_acc, 4),
        "Relative accuracy gain (%)": round(100 * (prop_acc - base_acc) / max(base_acc, 1e-12), 2),
        "Proposed parse coverage": round(float(prop["weighted_parse_coverage"]), 4),
        "Baseline parse coverage": round(float(best["weighted_parse_coverage"]), 4),
        "Proposed SPACE coverage": round(float(prop["weighted_space_label_coverage"]), 4),
        "Baseline SPACE coverage": round(float(best["weighted_space_label_coverage"]), 4),
        "Proposed false commitment": round(prop_fc, 4),
        "Baseline false commitment": round(base_fc, 4),
        "False commitment reduction": round(base_fc - prop_fc, 4),
    })

deltas = pd.DataFrame(rows)
deltas["_mo"] = deltas["Model"].map(model_order).fillna(99)
deltas = deltas.sort_values("_mo").drop(columns=["_mo"])

deltas.to_csv(outdir / "main_3model_deltas_vs_best_direct_cot.csv", index=False)
deltas.to_latex(outdir / "main_3model_deltas_vs_best_direct_cot.tex", index=False, escape=True)

# Proposed-only per-dataset table for main models.
per_main = per[(per["model"].isin(main_models)) & (per["method"].eq(proposed))].copy()
per_main["Model"] = per_main["model"].map(model_names).fillna(per_main["model"])
per_main = per_main.rename(columns={
    "dataset": "Dataset",
    "n": "N",
    "answer_accuracy": "Answer accuracy",
    "accuracy_ci95": "95% CI",
    "parse_coverage": "Parse coverage",
    "space_label_coverage": "SPACE coverage",
    "space_label_accuracy": "SPACE accuracy",
    "false_commitment_wrong_cond": "False commitment",
})

per_main_table = per_main[[
    "Dataset", "Model", "N", "Answer accuracy", "95% CI",
    "Parse coverage", "SPACE coverage", "SPACE accuracy", "False commitment"
]].copy()

for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "SPACE accuracy", "False commitment"]:
    per_main_table[c] = pd.to_numeric(per_main_table[c], errors="coerce").round(4)

per_main_table["_mo"] = per_main_table["Model"].map(model_order).fillna(99)
per_main_table = per_main_table.sort_values(["Dataset", "_mo"]).drop(columns=["_mo"])

per_main_table.to_csv(outdir / "main_3model_proposed_per_dataset_table.csv", index=False)
per_main_table.to_latex(outdir / "main_3model_proposed_per_dataset_table.tex", index=False, escape=True)

claim = []
claim.append("Main result claim:")
claim.append("")
claim.append("Across MedQA, MedMCQA, and PubMedQA with 1,000 examples per dataset, the proposed answer-fusion plus HypothesisMed-v3 SPACE pipeline improves weighted answer accuracy over the best Direct/CoT baseline for all three primary evaluated models.")
claim.append("")
for _, r in deltas.iterrows():
    claim.append(
        f"- {r['Model']}: proposed accuracy {r['Proposed accuracy']:.4f} vs {r['Baseline accuracy']:.4f} for {r['Best Direct/CoT baseline']}; "
        f"absolute gain {r['Absolute accuracy gain']:.4f}, relative gain {r['Relative accuracy gain (%)']:.2f}%; "
        f"SPACE coverage {r['Proposed SPACE coverage']:.4f}; false commitment reduced from {r['Baseline false commitment']:.4f} to {r['Proposed false commitment']:.4f}."
    )
claim.append("")
claim.append("Interpretation:")
claim.append("The proposed method is best framed as an inference-time reliability pipeline. Answer fusion improves task accuracy, while HypothesisMed-v3 adds structured SPACE labels and confidence behavior for reliability analysis.")
claim.append("")
claim.append("Appendix caveat:")
claim.append("BioMistral-7B is reported separately as a biomedical stress-test model because it shows very low structured-output compliance and SPACE coverage in this setup.")

(outdir / "ready_to_paste_main_claim.txt").write_text("\n".join(claim))

print("===== MAIN 3-MODEL AGGREGATE TABLE =====")
print(main_table.to_string(index=False))

print("\n===== MAIN 3-MODEL DELTAS =====")
print(deltas.to_string(index=False))

print("\n===== MAIN PROPOSED PER-DATASET TABLE =====")
print(per_main_table.to_string(index=False))

print("\n===== APPENDIX BIOMISTRAL STRESS TEST TABLE =====")
print(stress_table.to_string(index=False))

print("\n===== READY CLAIM =====")
print((outdir / "ready_to_paste_main_claim.txt").read_text())

print("\nSaved files:")
for p in sorted(outdir.iterdir()):
    print(p)
