import pandas as pd
import json
from pathlib import Path

outdir = Path("results/final_3model_paper")
outdir.mkdir(parents=True, exist_ok=True)

agg_path = Path("results/final_multimodel_extra/aggregate_all_models.csv")
per_path = Path("results/final_multimodel_extra/all_models_all_datasets_methods.csv")

agg = pd.read_csv(agg_path)
per = pd.read_csv(per_path)

proposed = "fusion_majority_answer_hypmed_v3_space"
baselines = ["direct", "cot", "hypothesismed_v3"]

# Keep only methods available across all 3 datasets.
main = agg[agg["datasets"].eq(3)].copy()

method_names = {
    "direct": "Direct",
    "cot": "Chain-of-thought",
    "hypothesismed_v3": "HypothesisMed-v3",
    "fusion_majority_answer_hypmed_v3_space": "Proposed: answer fusion + HypothesisMed-v3 SPACE",
}
model_names = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "deepseek_r1_qwen_32b": "DeepSeek-R1-Distill-Qwen-32B",
    "microsoft_phi_4_mini_instruct": "Phi-4-mini-instruct",
}

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

# Sort proposed first within each model, then CoT, Direct, HypothesisMed.
order = {
    "Proposed: answer fusion + HypothesisMed-v3 SPACE": 0,
    "Chain-of-thought": 1,
    "Direct": 2,
    "HypothesisMed-v3": 3,
}
main_table["_order"] = main_table["Method"].map(order).fillna(9)
main_table = main_table.sort_values(["Model", "_order"]).drop(columns=["_order"])

# Round numeric columns.
for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]:
    main_table[c] = main_table[c].astype(float).round(4)

main_table.to_csv(outdir / "main_3model_aggregate_table.csv", index=False)
main_table.to_latex(outdir / "main_3model_aggregate_table.tex", index=False, escape=True)

# Build deltas proposed vs best direct/CoT baseline for each model.
rows = []
for model in sorted(main["model"].unique()):
    m = main[main["model"].eq(model)]
    prop = m[m["method"].eq(proposed)]
    if prop.empty:
        continue
    prop = prop.iloc[0]

    base = m[m["method"].isin(["direct", "cot"])].copy()
    if base.empty:
        continue
    best = base.sort_values("weighted_answer_accuracy", ascending=False).iloc[0]

    rows.append({
        "model": model,
        "Model": model_names.get(model, model),
        "proposed_accuracy": round(float(prop["weighted_answer_accuracy"]), 4),
        "best_baseline_method": method_names.get(best["method"], best["method"]),
        "best_baseline_accuracy": round(float(best["weighted_answer_accuracy"]), 4),
        "absolute_accuracy_gain": round(float(prop["weighted_answer_accuracy"] - best["weighted_answer_accuracy"]), 4),
        "relative_accuracy_gain_percent": round(100.0 * float(prop["weighted_answer_accuracy"] - best["weighted_answer_accuracy"]) / max(float(best["weighted_answer_accuracy"]), 1e-12), 2),
        "proposed_parse_coverage": round(float(prop["weighted_parse_coverage"]), 4),
        "baseline_parse_coverage": round(float(best["weighted_parse_coverage"]), 4),
        "proposed_space_coverage": round(float(prop["weighted_space_label_coverage"]), 4),
        "baseline_space_coverage": round(float(best["weighted_space_label_coverage"]), 4),
        "proposed_false_commitment": round(float(prop["weighted_false_commitment_wrong_cond"]), 4),
        "baseline_false_commitment": round(float(best["weighted_false_commitment_wrong_cond"]), 4),
        "false_commitment_absolute_reduction": round(float(best["weighted_false_commitment_wrong_cond"] - prop["weighted_false_commitment_wrong_cond"]), 4),
    })

deltas = pd.DataFrame(rows)
deltas.to_csv(outdir / "deltas_vs_best_direct_cot_baseline.csv", index=False)
deltas.to_latex(outdir / "deltas_vs_best_direct_cot_baseline.tex", index=False, escape=True)

# Per-dataset compact table for proposed only and baselines.
per3 = per[per["method"].isin([proposed, "cot", "direct", "hypothesismed_v3"])].copy()
per3["Model"] = per3["model"].map(model_names).fillna(per3["model"])
per3["Method"] = per3["method"].map(method_names).fillna(per3["method"])
per3 = per3.rename(columns={
    "dataset": "Dataset",
    "answer_accuracy": "Answer accuracy",
    "parse_coverage": "Parse coverage",
    "space_label_coverage": "SPACE coverage",
    "false_commitment_wrong_cond": "False commitment",
})
compact_per = per3[[
    "Dataset", "Model", "Method", "n",
    "Answer accuracy", "accuracy_ci95",
    "Parse coverage", "SPACE coverage", "False commitment"
]].rename(columns={"n": "N", "accuracy_ci95": "95% CI"})

for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]:
    compact_per[c] = pd.to_numeric(compact_per[c], errors="coerce").round(4)

compact_per = compact_per.sort_values(["Dataset", "Model", "Method"])
compact_per.to_csv(outdir / "per_dataset_3model_table.csv", index=False)
compact_per.to_latex(outdir / "per_dataset_3model_table.tex", index=False, escape=True)

# Human-readable claim.
qwen = deltas[deltas["model"].eq("qwen2_5_7b_instruct")]
phi = deltas[deltas["model"].eq("microsoft_phi_4_mini_instruct")]
deep = deltas[deltas["model"].eq("deepseek_r1_qwen_32b")]

claim = []
claim.append("Main result claim for the 3-model paper:")
claim.append("")
claim.append("Across MedQA, MedMCQA, and PubMedQA with 1,000 examples per dataset, the proposed answer-fusion plus HypothesisMed-v3 SPACE pipeline improves weighted answer accuracy over the best Direct/CoT baseline for all three evaluated models.")
claim.append("")
for _, r in deltas.iterrows():
    claim.append(
        f"- {r['Model']}: proposed accuracy {r['proposed_accuracy']:.4f} vs {r['best_baseline_accuracy']:.4f} for {r['best_baseline_method']}; "
        f"absolute gain {r['absolute_accuracy_gain']:.4f}, relative gain {r['relative_accuracy_gain_percent']:.2f}%; "
        f"SPACE coverage {r['proposed_space_coverage']:.4f}; false commitment {r['proposed_false_commitment']:.4f}."
    )
claim.append("")
claim.append("Suggested framing:")
claim.append("HypothesisMed-v3 should be framed primarily as a reliability and structured reporting layer. Accuracy gains are obtained when this layer is paired with answer fusion. The strongest evidence is not that standalone prompting always improves raw accuracy, but that SPACE-aware fusion improves accuracy while producing near-complete structured reporting and lower false-commitment behavior.")
claim.append("")
claim.append("Important caveat:")
claim.append("DeepSeek-R1-Distill-Qwen-32B has low Direct/CoT parse coverage in this evaluation, so DeepSeek results should be interpreted as a formatting-stress test rather than as a clean head-to-head capability comparison.")

(outdir / "ready_to_paste_3model_claim.txt").write_text("\n".join(claim))

print("===== MAIN 3-MODEL AGGREGATE TABLE =====")
print(main_table.to_string(index=False))
print()
print("===== DELTAS VS BEST DIRECT/COT BASELINE =====")
print(deltas.to_string(index=False))
print()
print("===== READY CLAIM =====")
print((outdir / "ready_to_paste_3model_claim.txt").read_text())
print()
print("Saved files:")
for p in sorted(outdir.iterdir()):
    print(p)
