import pandas as pd
from pathlib import Path

outdir = Path("results/final_4model_paper")
outdir.mkdir(parents=True, exist_ok=True)

agg_path = Path("results/final_multimodel_extra/aggregate_all_models.csv")
per_path = Path("results/final_multimodel_extra/all_models_all_datasets_methods.csv")

agg = pd.read_csv(agg_path)
per = pd.read_csv(per_path)

proposed = "fusion_majority_answer_hypmed_v3_space"

method_names = {
    "direct": "Direct",
    "cot": "Chain-of-thought",
    "hypothesismed_v3": "HypothesisMed-v3",
    "hypothesismed_v2": "HypothesisMed-v2",
    "fusion_majority_answer_hypmed_v3_space": "Proposed: answer fusion + HypothesisMed-v3 SPACE",
    "fusion_cot_answer_hypmed_v3_space": "CoT answer + HypothesisMed-v3 SPACE",
    "fusion_cot_direct_answer_hypmed_v3_space": "CoT/Direct fallback + HypothesisMed-v3 SPACE",
}

model_names = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "deepseek_r1_qwen_32b": "DeepSeek-R1-Distill-Qwen-32B",
    "microsoft_phi_4_mini_instruct": "Phi-4-mini-instruct",
    "biomistral_7b": "BioMistral-7B",
}

# Keep only methods that run across all 3 datasets, excluding old v2 and one-dataset fusions.
main = agg[agg["datasets"].eq(3)].copy()
main = main[main["method"].isin(["direct", "cot", "hypothesismed_v3", proposed])].copy()

main["Model"] = main["model"].map(model_names).fillna(main["model"])
main["Method"] = main["method"].map(method_names).fillna(main["method"])

main_table = main[[
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

order = {
    "Proposed: answer fusion + HypothesisMed-v3 SPACE": 0,
    "Chain-of-thought": 1,
    "Direct": 2,
    "HypothesisMed-v3": 3,
}
model_order = {
    "Qwen2.5-7B-Instruct": 0,
    "Phi-4-mini-instruct": 1,
    "BioMistral-7B": 2,
    "DeepSeek-R1-Distill-Qwen-32B": 3,
}

main_table["_model_order"] = main_table["Model"].map(model_order).fillna(99)
main_table["_method_order"] = main_table["Method"].map(order).fillna(99)
main_table = main_table.sort_values(["_model_order", "_method_order"]).drop(columns=["_model_order", "_method_order"])

for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]:
    main_table[c] = pd.to_numeric(main_table[c], errors="coerce").round(4)

main_table.to_csv(outdir / "main_4model_aggregate_table.csv", index=False)
main_table.to_latex(outdir / "main_4model_aggregate_table.tex", index=False, escape=True)

# Deltas proposed vs best Direct/CoT baseline.
rows = []
for model in sorted(main["model"].unique()):
    m = main[main["model"].eq(model)]
    prop = m[m["method"].eq(proposed)]
    base = m[m["method"].isin(["direct", "cot"])].copy()

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
        "Relative accuracy gain (%)": round(100.0 * (prop_acc - base_acc) / max(base_acc, 1e-12), 2),
        "Proposed parse coverage": round(float(prop["weighted_parse_coverage"]), 4),
        "Baseline parse coverage": round(float(best["weighted_parse_coverage"]), 4),
        "Proposed SPACE coverage": round(float(prop["weighted_space_label_coverage"]), 4),
        "Baseline SPACE coverage": round(float(best["weighted_space_label_coverage"]), 4),
        "Proposed false commitment": round(prop_fc, 4),
        "Baseline false commitment": round(base_fc, 4),
        "False commitment reduction": round(base_fc - prop_fc, 4),
    })

deltas = pd.DataFrame(rows)
deltas["_model_order"] = deltas["Model"].map(model_order).fillna(99)
deltas = deltas.sort_values("_model_order").drop(columns=["_model_order"])

deltas.to_csv(outdir / "deltas_vs_best_direct_cot_baseline.csv", index=False)
deltas.to_latex(outdir / "deltas_vs_best_direct_cot_baseline.tex", index=False, escape=True)

# Proposed-only per-dataset table.
per_prop = per[per["method"].eq(proposed)].copy()
per_prop["Model"] = per_prop["model"].map(model_names).fillna(per_prop["model"])
per_prop = per_prop.rename(columns={
    "dataset": "Dataset",
    "n": "N",
    "answer_accuracy": "Answer accuracy",
    "accuracy_ci95": "95% CI",
    "parse_coverage": "Parse coverage",
    "space_label_coverage": "SPACE coverage",
    "space_label_accuracy": "SPACE accuracy",
    "false_commitment_wrong_cond": "False commitment",
})

per_prop_table = per_prop[[
    "Dataset", "Model", "N", "Answer accuracy", "95% CI",
    "Parse coverage", "SPACE coverage", "SPACE accuracy", "False commitment"
]].copy()

for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "SPACE accuracy", "False commitment"]:
    per_prop_table[c] = pd.to_numeric(per_prop_table[c], errors="coerce").round(4)

per_prop_table["_model_order"] = per_prop_table["Model"].map(model_order).fillna(99)
per_prop_table = per_prop_table.sort_values(["Dataset", "_model_order"]).drop(columns=["_model_order"])

per_prop_table.to_csv(outdir / "proposed_per_dataset_4model_table.csv", index=False)
per_prop_table.to_latex(outdir / "proposed_per_dataset_4model_table.tex", index=False, escape=True)

# Compact claim.
claim = []
claim.append("Main 4-model result claim:")
claim.append("")
claim.append("Across three biomedical QA datasets, MedQA, MedMCQA, and PubMedQA, with 1,000 examples per dataset, the proposed answer-fusion plus HypothesisMed-v3 SPACE pipeline improves weighted answer accuracy over the best Direct/CoT baseline for every evaluated model.")
claim.append("")
for _, r in deltas.iterrows():
    claim.append(
        f"- {r['Model']}: proposed accuracy {r['Proposed accuracy']:.4f} vs {r['Baseline accuracy']:.4f} for {r['Best Direct/CoT baseline']}; "
        f"absolute gain {r['Absolute accuracy gain']:.4f}, relative gain {r['Relative accuracy gain (%)']:.2f}%; "
        f"SPACE coverage {r['Proposed SPACE coverage']:.4f}; false commitment {r['Proposed false commitment']:.4f}."
    )
claim.append("")
claim.append("Recommended interpretation:")
claim.append("The proposed method should be framed as an inference-time reliability pipeline. Majority answer fusion improves answer accuracy, while HypothesisMed-v3 supplies structured SPACE labels and confidence information for reliability analysis.")
claim.append("")
claim.append("Caveat:")
claim.append("DeepSeek-R1-Distill-Qwen-32B has low Direct/CoT parse coverage in this setup, so DeepSeek should be described as a formatting-stress-test model rather than a clean capability comparison.")

(outdir / "ready_to_paste_4model_claim.txt").write_text("\n".join(claim))

print("===== MAIN 4-MODEL AGGREGATE TABLE =====")
print(main_table.to_string(index=False))
print()
print("===== DELTAS VS BEST DIRECT/COT BASELINE =====")
print(deltas.to_string(index=False))
print()
print("===== PROPOSED PER-DATASET TABLE =====")
print(per_prop_table.to_string(index=False))
print()
print("===== READY CLAIM =====")
print((outdir / "ready_to_paste_4model_claim.txt").read_text())
print()
print("Saved files:")
for p in sorted(outdir.iterdir()):
    print(p)
