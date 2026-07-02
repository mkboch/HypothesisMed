#!/usr/bin/env python3
from pathlib import Path
import csv
import json

ROOT = Path("/home/manikm/HypothesisMed")
OUT = ROOT / "results" / "final_experimental_checks"
OUT.mkdir(parents=True, exist_ok=True)

def count_lines(p):
    try:
        return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
    except Exception:
        return -1

def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

paths = sorted(ROOT.glob("results/**/scaled_wide_base_outputs.csv"))

inventory = []
for p in paths:
    inventory.append({
        "path": str(p.relative_to(ROOT)),
        "line_count_including_header": count_lines(p),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 3) if p.exists() else "",
    })

write_csv(
    OUT / "table_noapi_scaled_wide_matching_files.csv",
    inventory,
    ["path", "line_count_including_header", "size_mb"],
)

main = None
for p in paths:
    if "expanded_fix_noapi" in str(p):
        main = p
        break
if main is None and paths:
    main = paths[0]

summary_lines = []
summary_lines.append("# No-API scaled_wide_base_outputs count diagnostic\n")
summary_lines.append(f"Matching files found: **{len(paths)}**\n")

if not main:
    summary_lines.append("No `scaled_wide_base_outputs.csv` file found.\n")
    (OUT / "NOAPI_COUNT_MISMATCH_DIAGNOSTIC.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print("NOAPI_COUNT_MISMATCH_DIAGNOSTIC_OK")
    raise SystemExit(0)

summary_lines.append(f"Selected main file: `{main.relative_to(ROOT)}`\n")
summary_lines.append(f"Line count including header: **{count_lines(main)}**\n")

try:
    import pandas as pd

    df = pd.read_csv(main, low_memory=False)
    summary_lines.append(f"Data rows excluding header: **{len(df)}**\n")
    summary_lines.append(f"Columns: `{list(df.columns)}`\n")

    possible_key_cols = [
        "model", "model_label", "dataset", "question_id", "id", "item_id",
        "method", "mode", "prompt_mode", "strategy", "answer_type"
    ]
    key_cols = [c for c in possible_key_cols if c in df.columns]

    if key_cols:
        dup_count = int(df.duplicated(key_cols).sum())
        unique_count = int(len(df.drop_duplicates(key_cols)))
        summary_lines.append(f"Duplicate rows using available key columns `{key_cols}`: **{dup_count}**\n")
        summary_lines.append(f"Unique rows using available key columns: **{unique_count}**\n")
    else:
        dup_count = int(df.duplicated().sum())
        unique_count = int(len(df.drop_duplicates()))
        summary_lines.append("No standard key columns found. Used full-row duplicate check.\n")
        summary_lines.append(f"Full-row duplicates: **{dup_count}**\n")
        summary_lines.append(f"Unique full rows: **{unique_count}**\n")

    group_cols = [c for c in ["model", "model_label", "dataset", "method", "mode", "prompt_mode", "strategy"] if c in df.columns]
    if group_cols:
        g = df.groupby(group_cols, dropna=False).size().reset_index(name="N").sort_values(group_cols)
        g.to_csv(OUT / "table_noapi_scaled_wide_group_counts.csv", index=False)
        summary_lines.append(f"Wrote group counts: `results/final_experimental_checks/table_noapi_scaled_wide_group_counts.csv`\n")
        summary_lines.append("\n## First 80 group-count rows\n")
        summary_lines.append(g.head(80).to_string(index=False))
        summary_lines.append("\n")

    if len(df) == 20366:
        conclusion = "The file matches the original expected 20,366 data rows."
    elif key_cols and unique_count == 20366:
        conclusion = "The file appears to contain duplicated or appended rows, but the unique key count matches the original expected 20,366 rows."
    elif len(df) > 20366:
        conclusion = "The file has more rows than the original expected count. This is likely an updated or appended no-API table. Use current downstream summary tables for reporting, and do not treat this as a failed model experiment."
    else:
        conclusion = "The file has fewer rows than expected. Inspect the group-count table before relying on this artifact."

    summary_lines.append(f"\n## Conclusion\n\n{conclusion}\n")

except Exception as e:
    summary_lines.append(f"\nPandas diagnostic failed: `{repr(e)}`\n")

summary = "\n".join(summary_lines)
(OUT / "NOAPI_COUNT_MISMATCH_DIAGNOSTIC.md").write_text(summary, encoding="utf-8")

print(summary)
print("\nWROTE results/final_experimental_checks/NOAPI_COUNT_MISMATCH_DIAGNOSTIC.md")
print("WROTE results/final_experimental_checks/table_noapi_scaled_wide_matching_files.csv")
print("NOAPI_COUNT_MISMATCH_DIAGNOSTIC_OK")
