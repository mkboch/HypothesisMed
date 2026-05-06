from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile

ROOT = Path("/home/manikm/HypothesisMed")
ASSET = ROOT / "results/final_png_only_assets"
FIG = ASSET / "figures"
TAB = ASSET / "tables"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.8,
    "lines.markersize": 10,
})

def standardize_columns(df):
    rename = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        if key in ["model", "model_name"]:
            rename[c] = "Model"
        elif key in ["method"]:
            rename[c] = "Method"
        elif key in ["dataset", "datasets_name"]:
            rename[c] = "Dataset"
        elif key in ["datasets"]:
            rename[c] = "Datasets"
        elif key in ["n", "total_n"]:
            rename[c] = "N"
        elif key in ["answer_accuracy", "accuracy", "weighted_answer_accuracy"]:
            rename[c] = "Answer accuracy"
        elif key in ["parse_coverage", "weighted_parse_coverage"]:
            rename[c] = "Parse coverage"
        elif key in ["space_coverage", "space_label_coverage", "weighted_space_label_coverage"]:
            rename[c] = "SPACE coverage"
        elif key in ["space_accuracy", "space_label_accuracy"]:
            rename[c] = "SPACE accuracy"
        elif key in ["false_commitment", "false_commitment_wrong_cond", "weighted_false_commitment_wrong_cond"]:
            rename[c] = "False commitment"
        elif key in ["ci95", "accuracy_ci95", "95%_ci"]:
            rename[c] = "95% CI"
        elif key in ["proposed_accuracy"]:
            rename[c] = "Proposed accuracy"
        elif key in ["baseline_accuracy"]:
            rename[c] = "Baseline accuracy"
        elif key in ["absolute_accuracy_gain"]:
            rename[c] = "Absolute accuracy gain"
        elif key in ["relative_accuracy_gain_percent", "relative_accuracy_gain_(%)"]:
            rename[c] = "Relative accuracy gain (%)"
        elif key in ["proposed_parse_coverage"]:
            rename[c] = "Proposed parse coverage"
        elif key in ["baseline_parse_coverage"]:
            rename[c] = "Baseline parse coverage"
        elif key in ["proposed_space_coverage"]:
            rename[c] = "Proposed SPACE coverage"
        elif key in ["baseline_space_coverage"]:
            rename[c] = "Baseline SPACE coverage"
        elif key in ["proposed_false_commitment"]:
            rename[c] = "Proposed false commitment"
        elif key in ["baseline_false_commitment"]:
            rename[c] = "Baseline false commitment"
        elif key in ["false_commitment_reduction"]:
            rename[c] = "False commitment reduction"
        elif key in ["baseline_name", "best_direct/cot_baseline"]:
            rename[c] = "Best baseline"
    return df.rename(columns=rename)

def clean_model(x):
    x = str(x)
    repl = {
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "qwen2_5_7b_instruct": "Qwen2.5-7B",
        "Phi-4-mini-instruct": "Phi-4-mini",
        "microsoft_phi_4_mini_instruct": "Phi-4-mini",
        "DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-32B",
        "deepseek_r1_qwen_32b": "DeepSeek-R1-32B",
        "biomistral_biomistral_7b": "BioMistral-7B",
        "BioMistral/BioMistral-7B": "BioMistral-7B",
    }
    return repl.get(x, x)

def clean_method(x):
    x = str(x)
    repl = {
        "Proposed: answer fusion + HypothesisMed-v3 SPACE": "Proposed",
        "fusion_majority_answer_hypmed_v3_space": "Proposed",
        "Chain-of-thought": "CoT",
        "cot": "CoT",
        "Direct": "Direct",
        "direct": "Direct",
        "HypothesisMed-v3": "HypMed-v3",
        "hypothesismed_v3": "HypMed-v3",
    }
    return repl.get(x, x)

def clean_dataset(x):
    x = str(x).lower()
    return {"medqa": "MedQA", "medmcqa": "MedMCQA", "pubmedqa": "PubMedQA"}.get(x, str(x))

def savefig(name):
    out = FIG / name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.22)
    plt.close()
    print("saved", out)

def need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"\nERROR: missing columns for {name}: {miss}")
        print("Available columns:", list(df.columns))
        raise SystemExit(1)

main = standardize_columns(pd.read_csv(TAB / "table_main_4model_aggregate_clean.csv"))
deltas = standardize_columns(pd.read_csv(TAB / "table_deltas_vs_baseline_clean.csv"))
per = standardize_columns(pd.read_csv(TAB / "table_proposed_per_dataset_clean.csv"))
conf_dataset = standardize_columns(pd.read_csv(TAB / "confidence_accuracy_by_dataset.csv"))

conf_rows_path = TAB / "confidence_rows_from_fusion_outputs.csv"
conf_rows = pd.read_csv(conf_rows_path) if conf_rows_path.exists() else pd.DataFrame()

need(main, ["Model", "Method", "Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"], "main")
need(deltas, ["Model", "Absolute accuracy gain", "False commitment reduction"], "deltas")
need(per, ["Dataset", "Model", "Answer accuracy", "False commitment"], "per-dataset")

main["Model_clean"] = main["Model"].map(clean_model)
main["Method_clean"] = main["Method"].map(clean_method)
per["Model_clean"] = per["Model"].map(clean_model)
per["Dataset_clean"] = per["Dataset"].map(clean_dataset)
deltas["Model_clean"] = deltas["Model"].map(clean_model)

# Fig 1
pivot = main.pivot(index="Model_clean", columns="Method_clean", values="Answer accuracy")
methods = [m for m in ["Proposed", "CoT", "Direct", "HypMed-v3"] if m in pivot.columns]
pivot = pivot[methods]
ax = pivot.plot(kind="bar", figsize=(15, 8), width=0.78)
ax.set_title("Aggregate answer accuracy by model and method", pad=16)
ax.set_xlabel("Model")
ax.set_ylabel("Answer accuracy")
ax.set_ylim(0, max(0.85, float(np.nanmax(pivot.values)) + 0.08))
ax.tick_params(axis="x", rotation=18)
ax.legend(title="Method", ncol=2, frameon=True)
ax.grid(axis="y", alpha=0.3)
savefig("fig1_aggregate_answer_accuracy_by_model_method.png")

# Fig 2
plt.figure(figsize=(12, 7))
x = deltas["Model_clean"]
y = pd.to_numeric(deltas["Absolute accuracy gain"], errors="coerce")
plt.bar(x, y)
plt.title("Accuracy gain vs. best Direct/CoT baseline", pad=16)
plt.xlabel("Model")
plt.ylabel("Absolute accuracy gain")
plt.xticks(rotation=18, ha="right")
plt.grid(axis="y", alpha=0.3)
for i, v in enumerate(y):
    plt.text(i, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=16)
savefig("fig2_accuracy_gain_vs_best_baseline.png")

# Fig 3
pivot = per.pivot(index="Dataset_clean", columns="Model_clean", values="Answer accuracy")
dataset_order = [d for d in ["MedMCQA", "MedQA", "PubMedQA"] if d in pivot.index]
pivot = pivot.loc[dataset_order]
ax = pivot.plot(kind="bar", figsize=(14, 8), width=0.78)
ax.set_title("Proposed-method accuracy by dataset and model", pad=16)
ax.set_xlabel("Dataset")
ax.set_ylabel("Answer accuracy")
ax.set_ylim(0, max(0.85, float(np.nanmax(pivot.values)) + 0.08))
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Model", ncol=2, frameon=True)
ax.grid(axis="y", alpha=0.3)
savefig("fig3_proposed_accuracy_by_dataset_model.png")

# Fig 4
plt.figure(figsize=(12, 8))
for method in main["Method_clean"].unique():
    sub = main[main["Method_clean"] == method]
    plt.scatter(sub["Parse coverage"], sub["SPACE coverage"], s=180, label=method)
    for _, r in sub.iterrows():
        if r["Method_clean"] == "Proposed":
            plt.text(r["Parse coverage"] + 0.012, r["SPACE coverage"] + 0.012, r["Model_clean"], fontsize=14)
plt.title("Parse coverage vs. SPACE coverage", pad=16)
plt.xlabel("Parse coverage")
plt.ylabel("SPACE coverage")
plt.xlim(-0.03, 1.06)
plt.ylim(-0.03, 1.06)
plt.grid(alpha=0.3)
plt.legend(title="Method", frameon=True)
savefig("fig4_parse_vs_space_coverage.png")

# Fig 5
plt.figure(figsize=(12, 8))
xs = pd.to_numeric(deltas["Absolute accuracy gain"], errors="coerce")
ys = pd.to_numeric(deltas["False commitment reduction"], errors="coerce")
plt.scatter(xs, ys, s=200)
for _, r in deltas.iterrows():
    plt.text(float(r["Absolute accuracy gain"]) + 0.004, float(r["False commitment reduction"]) + 0.015, r["Model_clean"], fontsize=16)
plt.title("Accuracy gain vs. false-commitment reduction", pad=16)
plt.xlabel("Absolute accuracy gain")
plt.ylabel("False-commitment reduction")
plt.grid(alpha=0.3)
savefig("fig5_accuracy_gain_vs_false_commitment_reduction.png")

# Fig 6
prop = main[main["Method_clean"] == "Proposed"].copy()
metrics = ["Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]
mat = prop.set_index("Model_clean")[metrics]
plt.figure(figsize=(13, 7.5))
im = plt.imshow(mat.values, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(metrics)), ["Accuracy", "Parse cov.", "SPACE cov.", "False commit."], rotation=18, ha="right")
plt.yticks(range(len(mat.index)), mat.index)
plt.title("Proposed-method metric matrix", pad=16)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(j, i, f"{mat.values[i,j]:.3f}", ha="center", va="center", fontsize=16)
savefig("fig6_proposed_metric_matrix.png")

# Fig 7
heat = per.pivot(index="Dataset_clean", columns="Model_clean", values="Answer accuracy").loc[dataset_order]
plt.figure(figsize=(13, 7))
im = plt.imshow(heat.values, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(heat.columns)), heat.columns, rotation=18, ha="right")
plt.yticks(range(len(heat.index)), heat.index)
plt.title("Proposed-method accuracy heatmap", pad=16)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        plt.text(j, i, f"{heat.values[i,j]:.3f}", ha="center", va="center", fontsize=16)
savefig("fig7_proposed_accuracy_heatmap.png")

# Fig 8
pivot = main.pivot(index="Model_clean", columns="Method_clean", values="False commitment")
methods = [m for m in ["Proposed", "CoT", "Direct", "HypMed-v3"] if m in pivot.columns]
pivot = pivot[methods]
ax = pivot.plot(kind="bar", figsize=(15, 8), width=0.78)
ax.set_title("False commitment by model and method", pad=16)
ax.set_xlabel("Model")
ax.set_ylabel("False commitment")
ax.set_ylim(0, 1.08)
ax.tick_params(axis="x", rotation=18)
ax.legend(title="Method", ncol=2, frameon=True)
ax.grid(axis="y", alpha=0.3)
savefig("fig8_false_commitment_by_model_method.png")

# Fig 9
pivot = per.pivot(index="Dataset_clean", columns="Model_clean", values="False commitment").loc[dataset_order]
ax = pivot.plot(kind="bar", figsize=(14, 8), width=0.78)
ax.set_title("Proposed-method false commitment by dataset and model", pad=16)
ax.set_xlabel("Dataset")
ax.set_ylabel("False commitment")
ax.set_ylim(0, max(0.75, float(np.nanmax(pivot.values)) + 0.08))
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Model", ncol=2, frameon=True)
ax.grid(axis="y", alpha=0.3)
savefig("fig9_proposed_false_commitment_by_dataset_model.png")

# Fig 10
prop = main[main["Method_clean"] == "Proposed"].copy()
plt.figure(figsize=(12, 8))
plt.scatter(prop["Answer accuracy"], prop["False commitment"], s=200)
for _, r in prop.iterrows():
    plt.text(r["Answer accuracy"] + 0.008, r["False commitment"] + 0.015, r["Model_clean"], fontsize=16)
plt.title("Proposed accuracy vs. false commitment", pad=16)
plt.xlabel("Answer accuracy")
plt.ylabel("False commitment")
plt.grid(alpha=0.3)
savefig("fig10_proposed_accuracy_vs_false_commitment.png")

# Fig 11
plt.figure(figsize=(12, 7))
conf_col = None
for c in conf_rows.columns:
    if c.strip().lower() in ["confidence", "conf"]:
        conf_col = c
        break
if conf_col is not None and len(conf_rows) > 0:
    vals = pd.to_numeric(conf_rows[conf_col], errors="coerce").dropna()
    plt.hist(vals, bins=30)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence distribution for proposed fusion outputs", pad=16)
else:
    plt.text(0.5, 0.5, "Confidence rows unavailable", ha="center", va="center", fontsize=24)
    plt.axis("off")
savefig("fig11_confidence_distribution_proposed_fusion.png")

# Fig 12
cd = conf_dataset.copy()
dataset_col = None
mean_conf_col = None
acc_col = None
for c in cd.columns:
    lc = c.strip().lower()
    if lc == "dataset":
        dataset_col = c
    if "confidence" in lc and ("mean" in lc or "avg" in lc):
        mean_conf_col = c
    if "accuracy" in lc:
        acc_col = c
if dataset_col is None:
    dataset_col = cd.columns[0]
if mean_conf_col is None:
    for c in cd.columns:
        if "confidence" in c.lower():
            mean_conf_col = c
            break
if acc_col is None:
    for c in cd.columns:
        if "correct" in c.lower():
            acc_col = c
            break

plt.figure(figsize=(11, 7))
if mean_conf_col and acc_col:
    xs = pd.to_numeric(cd[mean_conf_col], errors="coerce")
    ys = pd.to_numeric(cd[acc_col], errors="coerce")
    plt.scatter(xs, ys, s=200)
    for i, r in cd.iterrows():
        plt.text(xs.iloc[i] + 0.004, ys.iloc[i] + 0.004, clean_dataset(r[dataset_col]), fontsize=17)
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Dataset-level confidence vs. accuracy", pad=16)
    plt.grid(alpha=0.3)
else:
    plt.text(0.5, 0.5, "Confidence summary columns unavailable", ha="center", va="center", fontsize=24)
    plt.axis("off")
savefig("fig12_dataset_confidence_vs_accuracy.png")

# Recreate zip
zip_path = ASSET / "HypothesisMed_png_only_assets.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted((ASSET / "figures").glob("*.png")):
        z.write(p, p.relative_to(ASSET))
    for p in sorted((ASSET / "tables").glob("*")):
        if p.is_file():
            z.write(p, p.relative_to(ASSET))
    readme = ASSET / "README_assets.txt"
    if readme.exists():
        z.write(readme, readme.relative_to(ASSET))

print("\nDONE. Regenerated readable PNG figures with same filenames.")
print("ZIP:", zip_path)
