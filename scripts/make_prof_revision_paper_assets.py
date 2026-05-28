#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import shutil

ROOT = Path("/home/manikm/HypothesisMed")
SRC = ROOT / "results" / "prof_revision_final"
OUT = SRC / "paper_assets"
FIG = OUT / "figures"
TEX = OUT / "tables"
CSV = OUT / "tables_csv_backup"

for d in [FIG, TEX, CSV]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ---------------------------------------------------------------------
# Copy and rename table assets
# ---------------------------------------------------------------------
table_map = {
    "scaled_compact_comparison": "table_scaled_compact_comparison",
    "expanded_space_stress_overall": "table_expanded_space_stress_overall",
    "expanded_space_stress_by_gold_label": "table_expanded_space_stress_by_gold_label",
    "scaled_fusion_calibration_summary": "table_scaled_fusion_calibration_summary",
    "scaled_two_model_weighted_aggregate": "table_scaled_two_model_weighted_aggregate",
    "scaled_mcnemar_exact_tests": "table_scaled_mcnemar_exact_tests",
    "scaled_fallback_order_sensitivity": "table_scaled_fallback_order_sensitivity",
    "structured_output_failure_taxonomy": "table_structured_output_failure_taxonomy",
    "expanded_space_stress_by_stress_type": "table_expanded_space_stress_by_stress_type",
    "expanded_space_stress_confusion_counts": "table_expanded_space_stress_confusion_counts",
    "scaled_fusion_calibration_bins": "table_scaled_fusion_calibration_bins",
}

for stem, newstem in table_map.items():
    for ext, destdir in [(".tex", TEX), (".csv", CSV)]:
        src = SRC / f"{stem}{ext}"
        if src.exists():
            shutil.copy2(src, destdir / f"{newstem}{ext}")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
compact = pd.read_csv(SRC / "scaled_compact_comparison.csv")
agg = pd.read_csv(SRC / "scaled_two_model_weighted_aggregate.csv")
stress_label = pd.read_csv(SRC / "expanded_space_stress_by_gold_label.csv")
cal = pd.read_csv(SRC / "scaled_fusion_calibration_summary.csv")
fail = pd.read_csv(SRC / "structured_output_failure_taxonomy.csv")

# ---------------------------------------------------------------------
# Figure 13: scaled accuracy comparison
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.2))
x = range(len(compact))
width = 0.35
ax.bar([i - width/2 for i in x], compact["Best baseline acc."], width, label="Best Direct/CoT")
ax.bar([i + width/2 for i in x], compact["Fusion acc."], width, label="Fusion")
ax.set_xticks(list(x))
ax.set_xticklabels(compact["Model"])
ax.set_ylim(0, max(compact["Best baseline acc."].max(), compact["Fusion acc."].max()) + 0.12)
ax.set_ylabel("Weighted accuracy")
ax.set_title("Scaled accuracy: fusion versus best answer-only baseline")
ax.legend(frameon=False)
for i, row in compact.iterrows():
    ax.text(i + width/2, row["Fusion acc."] + 0.012, f'{row["Fusion acc."]:.3f}', ha="center", va="bottom")
    ax.text(i - width/2, row["Best baseline acc."] + 0.012, f'{row["Best baseline acc."]:.3f}', ha="center", va="bottom")
fig.tight_layout()
fig.savefig(FIG / "fig13_scaled_accuracy_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 14: SPACE stress by gold label
# ---------------------------------------------------------------------
pivot = stress_label.pivot(index="Gold SPACE", columns="Model", values="Accuracy")
labels = list(pivot.index)
models = list(pivot.columns)
fig, ax = plt.subplots(figsize=(9.2, 5.4))
x = range(len(labels))
width = 0.35
for j, model in enumerate(models):
    vals = pivot[model].values
    pos = [i + (j - 0.5) * width for i in x]
    ax.bar(pos, vals, width, label=model)
    for px, v in zip(pos, vals):
        ax.text(px, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.75)
ax.set_ylabel("SPACE accuracy")
ax.set_title("Expanded SPACE stress test by gold label")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig14_expanded_space_stress_by_label.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 15: calibration ECE
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.4))
cal2 = cal.copy()
cal2["Group"] = cal2["Model"] + "\n" + cal2["Dataset"]
x = range(len(cal2))
ax.bar(list(x), cal2["ECE-10"])
ax.set_xticks(list(x))
ax.set_xticklabels(cal2["Group"], rotation=30, ha="right")
ax.set_ylabel("ECE-10")
ax.set_title("Scaled fusion calibration error")
for i, v in enumerate(cal2["ECE-10"]):
    ax.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig15_scaled_calibration_ece.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 16: structured-output failure rates for fusion only
# ---------------------------------------------------------------------
fusion_fail = fail[fail["Method"] == "Fusion"].copy()
fusion_fail["Answer missing rate"] = fusion_fail["Answer missing rate"].astype(float)
fusion_fail["SPACE missing rate"] = fusion_fail["SPACE missing rate"].astype(float)
fusion_fail["Multiple JSON rate"] = fusion_fail["Multiple JSON rate"].astype(float)
fusion_fail["Group"] = fusion_fail["Model"] + "\n" + fusion_fail["Dataset"]

fig, ax = plt.subplots(figsize=(10.5, 5.8))
x = range(len(fusion_fail))
width = 0.25
metrics = ["Answer missing rate", "SPACE missing rate", "Multiple JSON rate"]
for j, m in enumerate(metrics):
    pos = [i + (j - 1) * width for i in x]
    ax.bar(pos, fusion_fail[m], width, label=m)
ax.set_xticks(list(x))
ax.set_xticklabels(fusion_fail["Group"], rotation=30, ha="right")
ax.set_ylabel("Rate")
ax.set_title("Structured-output failure modes for fusion outputs")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig16_structured_output_failure_rates.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------
manifest = OUT / "README_prof_revision_assets.txt"
manifest.write_text(
    """Professor-revision paper assets.

Main-paper recommended tables:
- table_scaled_compact_comparison.tex
- table_expanded_space_stress_overall.tex
- table_expanded_space_stress_by_gold_label.tex
- table_scaled_fusion_calibration_summary.tex

Main-paper recommended figures:
- fig13_scaled_accuracy_comparison.png
- fig14_expanded_space_stress_by_label.png
- fig15_scaled_calibration_ece.png

Appendix recommended tables:
- table_scaled_two_model_weighted_aggregate.tex
- table_scaled_mcnemar_exact_tests.tex
- table_scaled_fallback_order_sensitivity.tex
- table_structured_output_failure_taxonomy.tex
- table_expanded_space_stress_by_stress_type.tex

Appendix recommended figure:
- fig16_structured_output_failure_rates.png
""",
    encoding="utf-8"
)

print("===== PAPER ASSETS CREATED =====")
print(f"Output folder: {OUT}")
print()
print("Figures:")
for p in sorted(FIG.glob("*.png")):
    print(p)
print()
print("Tables:")
for p in sorted(TEX.glob("*.tex")):
    print(p)
print()
print("CSV backups:")
for p in sorted(CSV.glob("*.csv")):
    print(p)
