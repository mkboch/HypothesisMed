import pandas as pd
from pathlib import Path

outdir = Path("results/final_claim_validation")
outdir.mkdir(parents=True, exist_ok=True)

# Your Qwen proposed per-dataset results from final table
ours = pd.DataFrame([
    {"dataset": "MedQA", "model": "Qwen2.5-7B-Instruct", "setting": "Ours: answer fusion + HypothesisMed-v3 SPACE", "accuracy": 0.596},
    {"dataset": "MedMCQA", "model": "Qwen2.5-7B-Instruct", "setting": "Ours: answer fusion + HypothesisMed-v3 SPACE", "accuracy": 0.561},
    {"dataset": "PubMedQA", "model": "Qwen2.5-7B-Instruct", "setting": "Ours: answer fusion + HypothesisMed-v3 SPACE", "accuracy": 0.753},
])

# Published values manually entered from papers you will cite.
# Important: these are not direct apples-to-apples unless the paper used same split and protocol.
lit = pd.DataFrame([
    {"dataset": "MedQA", "model": "Qwen2.5-7B-Instruct", "setting": "Published Qwen2.5-7B CoT baseline", "accuracy": 0.557, "source_note": "Disentangling Reasoning and Knowledge in Medical LLMs, Table S4"},
    {"dataset": "MedMCQA", "model": "Qwen2.5-7B-Instruct", "setting": "Published Qwen2.5-7B CoT baseline", "accuracy": 0.554, "source_note": "Disentangling Reasoning and Knowledge in Medical LLMs, Table S4"},
    {"dataset": "PubMedQA", "model": "Qwen2.5-7B-Instruct", "setting": "Published Qwen2.5-7B CoT baseline", "accuracy": 0.760, "source_note": "Disentangling Reasoning and Knowledge in Medical LLMs, Table S4"},

    {"dataset": "MedQA", "model": "Qwen2.5-7B-Instruct", "setting": "Published Qwen2.5 base", "accuracy": 0.570, "source_note": "MedReflect, Table 2"},
    {"dataset": "PubMedQA", "model": "Qwen2.5-7B-Instruct", "setting": "Published Qwen2.5 base", "accuracy": 0.556, "source_note": "MedReflect, Table 2"},

    {"dataset": "MedQA", "model": "BioMistral-7B", "setting": "Published BioMistral-7B", "accuracy": 0.444, "source_note": "BioMistral paper"},
    {"dataset": "MedMCQA", "model": "BioMistral-7B", "setting": "Published BioMistral-7B", "accuracy": 0.439, "source_note": "BioMistral paper"},
    {"dataset": "PubMedQA", "model": "BioMistral-7B", "setting": "Published BioMistral-7B", "accuracy": 0.376, "source_note": "BioMistral paper"},
])

combined = pd.concat([ours.assign(source_note="This work"), lit], ignore_index=True)
combined.to_csv(outdir / "literature_positioning_table.csv", index=False)

# Compare our Qwen proposed against published Qwen CoT values where available.
qwen_lit = lit[lit["setting"] == "Published Qwen2.5-7B CoT baseline"][["dataset", "accuracy"]].rename(columns={"accuracy": "published_qwen_cot"})
qwen_ours = ours[["dataset", "accuracy"]].rename(columns={"accuracy": "ours_proposed"})
cmp = qwen_ours.merge(qwen_lit, on="dataset", how="inner")
cmp["absolute_difference_ours_minus_published"] = cmp["ours_proposed"] - cmp["published_qwen_cot"]
cmp.loc["weighted_average"] = [
    "Average over available datasets",
    cmp["ours_proposed"].mean(),
    cmp["published_qwen_cot"].mean(),
    cmp["absolute_difference_ours_minus_published"].mean()
]
cmp.to_csv(outdir / "qwen_positioning_vs_published_cot.csv", index=False)

claim = """Recommended paper framing:

We do not claim a universal state-of-the-art result because published medical QA papers use different splits, prompts, shots, and sometimes fine-tuning. Instead, we claim that the proposed inference-time reliability pipeline is competitive with reported Qwen2.5-7B CoT performance on standard biomedical QA datasets while adding structured SPACE-label reporting, parseability, and false-commitment analysis.

Main claim:
Across MedQA, MedMCQA, and PubMedQA, answer fusion plus HypothesisMed-v3 SPACE improves each evaluated model over its own Direct/CoT baseline and reveals that answer accuracy, instruction-following, and structured reliability reporting are separable capabilities.

Caveat:
BioMistral-7B should be treated as a structured-output stress-test model, not as a primary capability comparison, because its published standard QA performance is substantially higher than its structured-output compliance in our setting.
"""
(outdir / "recommended_claim_framing.txt").write_text(claim)

print("\n===== Literature positioning table =====")
print(combined.to_string(index=False))

print("\n===== Qwen comparison against published CoT values =====")
print(cmp.to_string(index=False))

print("\nSaved:")
for p in sorted(outdir.glob("*")):
    print(p)
