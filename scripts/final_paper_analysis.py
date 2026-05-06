import json
import math
from pathlib import Path
import pandas as pd

SUMMARY = Path("results/paper_ready_fusion/medqa_original1000_fusion_summary.csv")
OUTDIR = Path("results/final_paper")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUMMARY)

# Keep only Qwen2.5 rows for main table
main = df[
    df["model"].eq("qwen2_5_7b_instruct")
].copy()

# Cleaner method names
rename = {
    "direct": "Direct",
    "cot": "CoT",
    "hypothesismed_v2": "HypothesisMed-v2",
    "hypothesismed_v3": "HypothesisMed-v3",
    "fusion_cot_answer_hypmed_v3_space": "CoT answer + HypothesisMed-v3 SPACE",
    "fusion_cot_direct_answer_hypmed_v3_space": "CoT/Direct fallback + HypothesisMed-v3 SPACE",
    "fusion_majority_answer_hypmed_v3_space": "Majority-answer fusion + HypothesisMed-v3 SPACE",
}
main["method_clean"] = main["method"].map(rename).fillna(main["method"])

main = main[[
    "method_clean",
    "n",
    "answer_accuracy",
    "accuracy_ci95",
    "parse_coverage",
    "space_label_coverage",
    "space_label_accuracy",
    "false_commitment_wrong_cond",
    "file"
]].sort_values("answer_accuracy", ascending=False)

main.to_csv(OUTDIR / "main_medqa_results_clean.csv", index=False)
main.to_latex(OUTDIR / "main_medqa_results_clean.tex", index=False, escape=True)

# Deltas versus CoT baseline
cot = df[(df["model"].eq("qwen2_5_7b_instruct")) & (df["method"].eq("cot"))].iloc[0]
best = df[(df["model"].eq("qwen2_5_7b_instruct")) & (df["method"].eq("fusion_majority_answer_hypmed_v3_space"))].iloc[0]

deltas = {
    "baseline": "CoT",
    "proposed": "Majority-answer fusion + HypothesisMed-v3 SPACE",
    "accuracy_baseline": float(cot["answer_accuracy"]),
    "accuracy_proposed": float(best["answer_accuracy"]),
    "accuracy_absolute_gain": round(float(best["answer_accuracy"]) - float(cot["answer_accuracy"]), 4),
    "accuracy_relative_gain_percent": round(100 * (float(best["answer_accuracy"]) - float(cot["answer_accuracy"])) / float(cot["answer_accuracy"]), 2),
    "parse_coverage_baseline": float(cot["parse_coverage"]),
    "parse_coverage_proposed": float(best["parse_coverage"]),
    "parse_coverage_gain": round(float(best["parse_coverage"]) - float(cot["parse_coverage"]), 4),
    "space_coverage_baseline": float(cot["space_label_coverage"]),
    "space_coverage_proposed": float(best["space_label_coverage"]),
    "space_coverage_gain": round(float(best["space_label_coverage"]) - float(cot["space_label_coverage"]), 4),
    "false_commitment_baseline": float(cot["false_commitment_wrong_cond"]),
    "false_commitment_proposed": float(best["false_commitment_wrong_cond"]),
    "false_commitment_reduction_absolute": round(float(cot["false_commitment_wrong_cond"]) - float(best["false_commitment_wrong_cond"]), 4),
    "false_commitment_reduction_relative_percent": round(100 * (float(cot["false_commitment_wrong_cond"]) - float(best["false_commitment_wrong_cond"])) / float(cot["false_commitment_wrong_cond"]), 2),
}

with open(OUTDIR / "deltas_vs_cot.json", "w") as f:
    json.dump(deltas, f, indent=2)

# Exact paired comparison using sign/binomial test logic without scipy
def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

cot_rows = {r["id"]: r for r in load_rows(cot["file"])}
best_rows = {r["id"]: r for r in load_rows(best["file"])}

ids = sorted(set(cot_rows) & set(best_rows))
cot_correct = []
best_correct = []
for i in ids:
    cot_correct.append(cot_rows[i].get("pred_answer") == cot_rows[i].get("gold_answer"))
    best_correct.append(best_rows[i].get("pred_answer") == best_rows[i].get("gold_answer"))

best_only = sum((not c) and b for c, b in zip(cot_correct, best_correct))
cot_only = sum(c and (not b) for c, b in zip(cot_correct, best_correct))
both_correct = sum(c and b for c, b in zip(cot_correct, best_correct))
both_wrong = sum((not c) and (not b) for c, b in zip(cot_correct, best_correct))

# two-sided exact binomial p-value for discordant pairs under p=0.5
n_disc = best_only + cot_only
k = min(best_only, cot_only)

def comb(n, r):
    return math.comb(n, r)

if n_disc > 0:
    p_one_tail = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc)
    p_two = min(1.0, 2 * p_one_tail)
else:
    p_two = 1.0

paired = {
    "n_shared": len(ids),
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "proposed_correct_cot_wrong": best_only,
    "cot_correct_proposed_wrong": cot_only,
    "discordant_pairs": n_disc,
    "exact_two_sided_binomial_p": p_two,
}

with open(OUTDIR / "paired_comparison_vs_cot.json", "w") as f:
    json.dump(paired, f, indent=2)

claim = f"""Main result claim:

On 1,000 MedQA questions, the proposed majority-answer fusion with HypothesisMed-v3 SPACE reporting achieved {best['answer_accuracy']:.3f} answer accuracy, compared with {cot['answer_accuracy']:.3f} for the CoT baseline. This corresponds to an absolute gain of {deltas['accuracy_absolute_gain']:.3f} and a relative gain of {deltas['accuracy_relative_gain_percent']:.2f}%. The proposed method also increased parse coverage from {cot['parse_coverage']:.3f} to {best['parse_coverage']:.3f}, increased SPACE-label coverage from {cot['space_label_coverage']:.3f} to {best['space_label_coverage']:.3f}, and reduced high-confidence wrong commitments from {cot['false_commitment_wrong_cond']:.3f} to {best['false_commitment_wrong_cond']:.3f}.

Paired comparison versus CoT:
Proposed correct / CoT wrong: {best_only}
CoT correct / proposed wrong: {cot_only}
Exact two-sided binomial p-value over discordant pairs: {p_two:.6g}

Suggested interpretation:
The improvement should be framed as an inference-time reliability ensemble rather than as a standalone prompting gain. The answer majority vote improves raw accuracy, while HypothesisMed-v3 supplies structured SPACE labels and confidence behavior for reliability analysis.
"""

(OUTDIR / "ready_to_paste_claim.txt").write_text(claim)

print("===== CLEAN MAIN TABLE =====")
print(main.to_string(index=False))
print()
print("===== DELTAS VS COT =====")
print(json.dumps(deltas, indent=2))
print()
print("===== PAIRED COMPARISON VS COT =====")
print(json.dumps(paired, indent=2))
print()
print("===== READY CLAIM =====")
print(claim)
print()
print("Saved:")
print(OUTDIR / "main_medqa_results_clean.csv")
print(OUTDIR / "main_medqa_results_clean.tex")
print(OUTDIR / "deltas_vs_cot.json")
print(OUTDIR / "paired_comparison_vs_cot.json")
print(OUTDIR / "ready_to_paste_claim.txt")
