import os
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "results" / "final_png_only_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

main_csv = ROOT / "results/final_4model_paper/main_4model_aggregate_table.csv"
per_csv = ROOT / "results/final_4model_paper/proposed_per_dataset_4model_table.csv"
delta_csv = ROOT / "results/final_4model_paper/deltas_vs_best_direct_cot_baseline.csv"

if not main_csv.exists():
    raise FileNotFoundError(f"Missing {main_csv}")
if not per_csv.exists():
    raise FileNotFoundError(f"Missing {per_csv}")
if not delta_csv.exists():
    raise FileNotFoundError(f"Missing {delta_csv}")

main = pd.read_csv(main_csv)
per = pd.read_csv(per_csv)
delta = pd.read_csv(delta_csv)

# Normalize names if needed
main = main.rename(columns={
    "Answer accuracy": "answer_accuracy",
    "Parse coverage": "parse_coverage",
    "SPACE coverage": "space_coverage",
    "False commitment": "false_commitment",
})
per = per.rename(columns={
    "Dataset": "dataset",
    "Model": "model",
    "N": "n",
    "Answer accuracy": "answer_accuracy",
    "95% CI": "ci95",
    "Parse coverage": "parse_coverage",
    "SPACE coverage": "space_coverage",
    "SPACE accuracy": "space_accuracy",
    "False commitment": "false_commitment",
})
delta = delta.rename(columns={
    "Model": "model",
    "Proposed accuracy": "proposed_accuracy",
    "Best Direct/CoT baseline": "baseline_name",
    "Baseline accuracy": "baseline_accuracy",
    "Absolute accuracy gain": "absolute_accuracy_gain",
    "Relative accuracy gain (%)": "relative_accuracy_gain_percent",
    "Proposed parse coverage": "proposed_parse_coverage",
    "Baseline parse coverage": "baseline_parse_coverage",
    "Proposed SPACE coverage": "proposed_space_coverage",
    "Baseline SPACE coverage": "baseline_space_coverage",
    "Proposed false commitment": "proposed_false_commitment",
    "Baseline false commitment": "baseline_false_commitment",
    "False commitment reduction": "false_commitment_reduction",
})

# Clean display labels
def clean_model(x):
    x = str(x)
    return {
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "Phi-4-mini-instruct": "Phi-4-mini",
        "DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-32B",
        "biomistral_biomistral_7b": "BioMistral-7B",
        "BioMistral/BioMistral-7B": "BioMistral-7B",
    }.get(x, x)

def clean_method(x):
    x = str(x)
    return {
        "Proposed: answer fusion + HypothesisMed-v3 SPACE": "Proposed",
        "Chain-of-thought": "CoT",
        "Direct": "Direct",
        "HypothesisMed-v3": "HypMed-v3",
    }.get(x, x)

main["model_clean"] = main["Model"].map(clean_model)
main["method_clean"] = main["Method"].map(clean_method)
per["model_clean"] = per["model"].map(clean_model)
delta["model_clean"] = delta["model"].map(clean_model)

# Save cleaned tables
main.to_csv(TAB / "table_main_4model_aggregate_clean.csv", index=False)
per.to_csv(TAB / "table_proposed_per_dataset_clean.csv", index=False)
delta.to_csv(TAB / "table_deltas_vs_baseline_clean.csv", index=False)

def savefig(name):
    path = FIG / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

# -------------------------
# Figure 1: Aggregate answer accuracy by model and method
# -------------------------
pivot = main.pivot_table(index="model_clean", columns="method_clean", values="answer_accuracy", aggfunc="mean")
order_models = ["Qwen2.5-7B", "Phi-4-mini", "DeepSeek-R1-32B", "BioMistral-7B"]
order_methods = [m for m in ["Direct", "CoT", "HypMed-v3", "Proposed"] if m in pivot.columns]
pivot = pivot.reindex([m for m in order_models if m in pivot.index])[order_methods]

ax = pivot.plot(kind="bar", figsize=(10, 5), rot=20)
ax.set_title("Aggregate answer accuracy across three biomedical QA datasets")
ax.set_ylabel("Weighted answer accuracy")
ax.set_xlabel("Model")
ax.set_ylim(0, max(0.75, float(np.nanmax(pivot.values)) + 0.08))
ax.legend(title="Method", loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
savefig("fig1_aggregate_answer_accuracy_by_model_method")

# -------------------------
# Figure 2: Proposed accuracy gain over best Direct/CoT baseline
# -------------------------
d = delta.copy()
d = d.sort_values("absolute_accuracy_gain", ascending=False)
plt.figure(figsize=(9, 4.8))
bars = plt.bar(d["model_clean"], d["absolute_accuracy_gain"])
plt.axhline(0, linewidth=1)
plt.title("Absolute answer-accuracy gain over each model's best Direct/CoT baseline")
plt.ylabel("Absolute accuracy gain")
plt.xlabel("Model")
plt.xticks(rotation=20, ha="right")
for bar, val in zip(bars, d["absolute_accuracy_gain"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)
savefig("fig2_accuracy_gain_vs_best_baseline")

# -------------------------
# Figure 3: Proposed per-dataset accuracy by model
# -------------------------
p = per.copy()
piv = p.pivot_table(index="dataset", columns="model_clean", values="answer_accuracy", aggfunc="mean")
dataset_order = [x for x in ["medqa", "medmcqa", "pubmedqa"] if x in piv.index]
piv = piv.reindex(dataset_order)
piv = piv[[m for m in order_models if m in piv.columns]]

ax = piv.plot(kind="bar", figsize=(10, 5), rot=0)
ax.set_title("Proposed pipeline answer accuracy by dataset and model")
ax.set_ylabel("Answer accuracy")
ax.set_xlabel("Dataset")
ax.set_ylim(0, max(0.85, float(np.nanmax(piv.values)) + 0.08))
ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1))
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
savefig("fig3_proposed_accuracy_by_dataset_model")

# -------------------------
# Figure 4: Parse coverage vs SPACE coverage
# -------------------------
plt.figure(figsize=(8, 5.5))
for method, group in main.groupby("method_clean"):
    plt.scatter(group["parse_coverage"], group["space_coverage"], label=method, s=70)
    for _, r in group.iterrows():
        plt.text(r["parse_coverage"] + 0.005, r["space_coverage"] + 0.005, r["model_clean"], fontsize=8)
plt.title("Structured-output behavior: parse coverage versus SPACE coverage")
plt.xlabel("Parse coverage")
plt.ylabel("SPACE-label coverage")
plt.xlim(-0.03, 1.05)
plt.ylim(-0.03, 1.05)
plt.legend(title="Method", loc="lower right")
savefig("fig4_parse_vs_space_coverage")

# -------------------------
# Figure 5: Accuracy gain vs false-commitment reduction
# -------------------------
plt.figure(figsize=(8, 5))
plt.scatter(delta["absolute_accuracy_gain"], delta["false_commitment_reduction"], s=90)
for _, r in delta.iterrows():
    plt.text(r["absolute_accuracy_gain"] + 0.003, r["false_commitment_reduction"] + 0.01, r["model_clean"], fontsize=9)
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)
plt.title("Reliability tradeoff: accuracy gain and false-commitment reduction")
plt.xlabel("Absolute accuracy gain over best Direct/CoT baseline")
plt.ylabel("False-commitment reduction")
savefig("fig5_accuracy_gain_vs_false_commitment_reduction")

# -------------------------
# Figure 6: Proposed pipeline metric matrix by model
# -------------------------
prop = main[main["method_clean"] == "Proposed"].copy()
prop = prop.set_index("model_clean").reindex([m for m in order_models if m in prop["model_clean"].values])
metrics = ["answer_accuracy", "parse_coverage", "space_coverage", "false_commitment"]
mat = prop[metrics].astype(float)

plt.figure(figsize=(8.5, 4.8))
plt.imshow(mat.values, aspect="auto")
plt.colorbar(label="Metric value")
plt.xticks(range(len(metrics)), ["Accuracy", "Parse", "SPACE", "False commit"], rotation=20, ha="right")
plt.yticks(range(len(mat.index)), mat.index)
plt.title("Proposed pipeline metric matrix")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
savefig("fig6_proposed_metric_matrix")

# -------------------------
# Figure 7: Proposed model-dataset accuracy heatmap
# -------------------------
heat = per.pivot_table(index="model_clean", columns="dataset", values="answer_accuracy", aggfunc="mean")
heat = heat.reindex([m for m in order_models if m in heat.index])
heat = heat[[d for d in ["medqa", "medmcqa", "pubmedqa"] if d in heat.columns]]

plt.figure(figsize=(7.5, 4.8))
plt.imshow(heat.values, aspect="auto")
plt.colorbar(label="Answer accuracy")
plt.xticks(range(len(heat.columns)), heat.columns)
plt.yticks(range(len(heat.index)), heat.index)
plt.title("Proposed answer accuracy matrix across models and datasets")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        plt.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
savefig("fig7_proposed_accuracy_heatmap")

# -------------------------
# Figure 8: False commitment by method and model
# -------------------------
fc = main.pivot_table(index="model_clean", columns="method_clean", values="false_commitment", aggfunc="mean")
fc = fc.reindex([m for m in order_models if m in fc.index])
fc = fc[[m for m in ["Direct", "CoT", "HypMed-v3", "Proposed"] if m in fc.columns]]

ax = fc.plot(kind="bar", figsize=(10, 5), rot=20)
ax.set_title("False commitment rate by model and method")
ax.set_ylabel("False commitment")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.05)
ax.legend(title="Method", loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
savefig("fig8_false_commitment_by_model_method")

# -------------------------
# Figure 9: Per-dataset proposed false commitment
# -------------------------
fc2 = per.pivot_table(index="dataset", columns="model_clean", values="false_commitment", aggfunc="mean")
fc2 = fc2.reindex([x for x in ["medqa", "medmcqa", "pubmedqa"] if x in fc2.index])
fc2 = fc2[[m for m in order_models if m in fc2.columns]]

ax = fc2.plot(kind="bar", figsize=(10, 5), rot=0)
ax.set_title("Proposed pipeline false commitment by dataset and model")
ax.set_ylabel("False commitment")
ax.set_xlabel("Dataset")
ax.set_ylim(0, 1.05)
ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1))
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
savefig("fig9_proposed_false_commitment_by_dataset_model")

# -------------------------
# Figure 10: Accuracy vs false commitment for proposed pipeline
# -------------------------
plt.figure(figsize=(8, 5))
plt.scatter(prop["answer_accuracy"], prop["false_commitment"], s=90)
for _, r in prop.reset_index().iterrows():
    plt.text(r["answer_accuracy"] + 0.005, r["false_commitment"] + 0.01, r["model_clean"], fontsize=9)
plt.title("Proposed pipeline: answer accuracy versus false commitment")
plt.xlabel("Weighted answer accuracy")
plt.ylabel("False commitment")
plt.xlim(0, max(0.75, prop["answer_accuracy"].max() + 0.08))
plt.ylim(-0.03, 1.05)
savefig("fig10_proposed_accuracy_vs_false_commitment")

# -------------------------
# Optional confidence histograms from result JSONL files
# -------------------------
def read_jsonl(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception:
        return []
    return rows

fusion_files = sorted((ROOT / "results/fusion").glob("*fusion_majority_answer_hypmed_v3_space_*_original1000.jsonl"))
conf_rows = []
for fp in fusion_files:
    rows = read_jsonl(fp)
    for r in rows:
        c = r.get("confidence", None)
        if isinstance(c, (int, float)):
            conf_rows.append({
                "file": str(fp),
                "model": r.get("model", fp.name.split("_fusion_")[0]),
                "dataset": r.get("dataset", "unknown"),
                "confidence": float(c),
                "correct": int(str(r.get("pred_answer")) == str(r.get("gold_answer"))),
            })

if conf_rows:
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(TAB / "confidence_rows_from_fusion_outputs.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.hist(conf["confidence"].dropna(), bins=20)
    plt.title("Confidence distribution for proposed fusion outputs")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    savefig("fig11_confidence_distribution_proposed_fusion")

    agg_conf = conf.groupby("dataset").agg(
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
        n=("correct", "size")
    ).reset_index()
    agg_conf.to_csv(TAB / "confidence_accuracy_by_dataset.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(agg_conf["mean_confidence"], agg_conf["accuracy"], s=90)
    for _, r in agg_conf.iterrows():
        plt.text(r["mean_confidence"] + 0.005, r["accuracy"] + 0.005, r["dataset"], fontsize=9)
    plt.title("Dataset-level confidence versus accuracy")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    savefig("fig12_dataset_confidence_vs_accuracy")

# README
readme = OUT / "README_assets.txt"
readme.write_text(
    "HypothesisMed PNG-only final asset package\n\n"
    "Figures are stored in figures/*.png only.\n"
    "Tables are stored in tables/*.csv.\n\n"
    "Recommended main-paper figures:\n"
    "1. fig1_aggregate_answer_accuracy_by_model_method.png\n"
    "2. fig2_accuracy_gain_vs_best_baseline.png\n"
    "3. fig3_proposed_accuracy_by_dataset_model.png\n"
    "4. fig4_parse_vs_space_coverage.png\n"
    "5. fig5_accuracy_gain_vs_false_commitment_reduction.png\n\n"
    "Additional appendix/SI figures:\n"
    "fig6 to fig12, depending on manuscript space.\n",
    encoding="utf-8"
)

# Zip everything
zip_path = OUT / "HypothesisMed_png_only_assets.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob("*"):
        if path == zip_path:
            continue
        z.write(path, path.relative_to(OUT))

print("\n===== CREATED PNG FIGURES =====")
for pth in sorted(FIG.glob("*.png")):
    print(pth)

print("\n===== CREATED TABLES =====")
for pth in sorted(TAB.glob("*.csv")):
    print(pth)

print("\n===== ZIP READY =====")
print(zip_path)
