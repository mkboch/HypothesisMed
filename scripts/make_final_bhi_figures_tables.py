from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import shutil
import zipfile

ROOT = Path(".")
OUT = ROOT / "results" / "final_4model_paper_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

main_csv = ROOT / "results/final_4model_paper/main_4model_aggregate_table.csv"
delta_csv = ROOT / "results/final_4model_paper/deltas_vs_best_direct_cot_baseline.csv"
per_csv = ROOT / "results/final_4model_paper/proposed_per_dataset_4model_table.csv"

for p in [main_csv, delta_csv, per_csv]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

main = pd.read_csv(main_csv)
delta = pd.read_csv(delta_csv)
per = pd.read_csv(per_csv)

# Normalize display names if needed
model_name_map = {
    "biomistral_biomistral_7b": "BioMistral-7B",
    "BioMistral/BioMistral-7B": "BioMistral-7B",
}
for df in [main, delta, per]:
    if "Model" in df.columns:
        df["Model"] = df["Model"].replace(model_name_map)

# ---------- TABLES ----------
def save_table(df, name):
    csv_path = TAB / f"{name}.csv"
    tex_path = TAB / f"{name}.tex"
    xlsx_path = TAB / f"{name}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))
    return csv_path, tex_path, xlsx_path

# Table 1: compact aggregate
table1_cols = ["Model", "Method", "Datasets", "N", "Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]
table1 = main[table1_cols].copy()
for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "False commitment"]:
    table1[c] = table1[c].astype(float).round(4)
save_table(table1, "table1_main_4model_aggregate")

# Table 2: deltas
table2_cols = [
    "Model",
    "Proposed accuracy",
    "Best Direct/CoT baseline",
    "Baseline accuracy",
    "Absolute accuracy gain",
    "Relative accuracy gain (%)",
    "Proposed SPACE coverage",
    "Baseline SPACE coverage",
    "Proposed false commitment",
    "Baseline false commitment",
    "False commitment reduction",
]
table2 = delta[table2_cols].copy()
for c in table2.columns:
    if c not in ["Model", "Best Direct/CoT baseline"]:
        table2[c] = pd.to_numeric(table2[c], errors="coerce").round(4)
save_table(table2, "table2_deltas_vs_best_baseline")

# Table 3: proposed per-dataset
table3_cols = ["Dataset", "Model", "N", "Answer accuracy", "95% CI", "Parse coverage", "SPACE coverage", "SPACE accuracy", "False commitment"]
table3 = per[table3_cols].copy()
for c in ["Answer accuracy", "Parse coverage", "SPACE coverage", "SPACE accuracy", "False commitment"]:
    table3[c] = pd.to_numeric(table3[c], errors="coerce").round(4)
save_table(table3, "table3_proposed_per_dataset")

# ---------- FIGURE 1: workflow ----------
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.axis("off")

boxes = [
    ("Biomedical QA\nquestion + options", 0.05, 0.52),
    ("Direct\nanswer", 0.29, 0.75),
    ("CoT\nanswer", 0.29, 0.52),
    ("HypothesisMed-v3\nSPACE output", 0.29, 0.29),
    ("Answer fusion\nmajority vote", 0.56, 0.62),
    ("Structured reliability\nSPACE + confidence", 0.56, 0.34),
    ("Final output\nanswer + reliability artifacts", 0.82, 0.48),
]

for text, x, y in boxes:
    ax.text(
        x, y, text,
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", linewidth=1.2, facecolor="white")
    )

arrows = [
    ((0.16, 0.52), (0.23, 0.75)),
    ((0.16, 0.52), (0.23, 0.52)),
    ((0.16, 0.52), (0.23, 0.29)),
    ((0.39, 0.75), (0.49, 0.62)),
    ((0.39, 0.52), (0.49, 0.62)),
    ((0.39, 0.29), (0.49, 0.34)),
    ((0.66, 0.62), (0.75, 0.48)),
    ((0.66, 0.34), (0.75, 0.48)),
]
for (x1, y1), (x2, y2) in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3))

ax.set_title("HypothesisMed inference-time reliability pipeline", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(FIG / "fig1_hypothesismed_workflow.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig1_hypothesismed_workflow.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- FIGURE 2: aggregate accuracy ----------
plot_df = main[["Model", "Method", "Answer accuracy"]].copy()
plot_df["Answer accuracy"] = pd.to_numeric(plot_df["Answer accuracy"], errors="coerce")
preferred_order = [
    "Direct",
    "Chain-of-thought",
    "HypothesisMed-v3",
    "Proposed: answer fusion + HypothesisMed-v3 SPACE",
]
plot_df["Method"] = pd.Categorical(plot_df["Method"], preferred_order, ordered=True)
plot_df = plot_df.sort_values(["Model", "Method"])

models = list(plot_df["Model"].drop_duplicates())
methods = preferred_order
x = np.arange(len(models))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 5.2))
for i, m in enumerate(methods):
    vals = []
    for model in models:
        row = plot_df[(plot_df["Model"] == model) & (plot_df["Method"] == m)]
        vals.append(float(row["Answer accuracy"].iloc[0]) if len(row) else np.nan)
    ax.bar(x + (i - 1.5) * width, vals, width, label=m)

ax.set_ylabel("Weighted answer accuracy")
ax.set_ylim(0, 0.75)
ax.set_xticks(x)
ax.set_xticklabels([textwrap.fill(m, 18) for m in models], rotation=0)
ax.set_title("Aggregate biomedical QA accuracy across models and methods")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "fig2_aggregate_accuracy_by_model_method.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig2_aggregate_accuracy_by_model_method.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- FIGURE 3: reliability tradeoff ----------
rel = main[["Model", "Method", "SPACE coverage", "False commitment"]].copy()
rel["SPACE coverage"] = pd.to_numeric(rel["SPACE coverage"], errors="coerce")
rel["False commitment"] = pd.to_numeric(rel["False commitment"], errors="coerce")
rel = rel.dropna(subset=["SPACE coverage", "False commitment"])

fig, ax = plt.subplots(figsize=(8, 6))
for _, r in rel.iterrows():
    marker = "o"
    if "Proposed" in str(r["Method"]):
        marker = "s"
    ax.scatter(r["SPACE coverage"], r["False commitment"], marker=marker, s=75)
    label = f"{r['Model'].split('-')[0][:10]} / {str(r['Method']).replace('Proposed: answer fusion + HypothesisMed-v3 SPACE', 'Proposed').replace('Chain-of-thought', 'CoT')}"
    ax.annotate(label, (r["SPACE coverage"], r["False commitment"]), fontsize=7, xytext=(4, 3), textcoords="offset points")

ax.set_xlabel("SPACE-label coverage")
ax.set_ylabel("False commitment rate")
ax.set_xlim(-0.03, 1.05)
ax.set_ylim(-0.03, 1.05)
ax.set_title("Structured reliability tradeoff across models and methods")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "fig3_space_coverage_vs_false_commitment.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig3_space_coverage_vs_false_commitment.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Optional appendix figure: proposed per-dataset heatmap-like matrix ----------
pivot = per.pivot_table(index="Model", columns="Dataset", values="Answer accuracy", aggfunc="first")
pivot = pivot.apply(pd.to_numeric, errors="coerce")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
im = ax.imshow(pivot.values, aspect="auto")
ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_yticklabels([textwrap.fill(x, 24) for x in pivot.index])
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.3f}" if not np.isnan(val) else "", ha="center", va="center", fontsize=9)
ax.set_title("Proposed method accuracy by dataset and model")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG / "appendix_fig_proposed_accuracy_matrix.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "appendix_fig_proposed_accuracy_matrix.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Write figure/table index ----------
index = OUT / "README_assets.txt"
index.write_text(
"""HypothesisMed BHI paper assets

Recommended main figures:
1. figures/fig1_hypothesismed_workflow.png/pdf
2. figures/fig2_aggregate_accuracy_by_model_method.png/pdf
3. figures/fig3_space_coverage_vs_false_commitment.png/pdf

Optional appendix figure:
- figures/appendix_fig_proposed_accuracy_matrix.png/pdf

Recommended main tables:
1. tables/table1_main_4model_aggregate.csv/tex/xlsx
2. tables/table2_deltas_vs_best_baseline.csv/tex/xlsx
3. tables/table3_proposed_per_dataset.csv/tex/xlsx

Source CSVs:
- results/final_4model_paper/main_4model_aggregate_table.csv
- results/final_4model_paper/deltas_vs_best_direct_cot_baseline.csv
- results/final_4model_paper/proposed_per_dataset_4model_table.csv
"""
)

# ---------- Zip everything ----------
zip_path = OUT / "HypothesisMed_BHI_figures_tables.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p == zip_path:
            continue
        z.write(p, arcname=p.relative_to(OUT))

print("\nDONE. Created paper assets in:")
print(OUT.resolve())
print("\nFigures:")
for p in sorted(FIG.glob("*")):
    print(" -", p)
print("\nTables:")
for p in sorted(TAB.glob("*")):
    print(" -", p)
print("\nZIP:")
print(zip_path.resolve())
