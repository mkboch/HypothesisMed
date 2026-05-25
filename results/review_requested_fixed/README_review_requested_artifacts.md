# Reviewer-requested HypothesisMed artifacts

This folder contains additional analysis artifacts added for the revised HypothesisMed paper.

## Files

- `space_stress_compact_table.tex`: LaTeX table for the controlled SPACE stress-test.
- `space_stress_compact_table.csv`: CSV version of the controlled SPACE stress-test results.
- `mcnemar_fixed.csv`: Paired exact/McNemar-style tests comparing the proposed method with the best Direct/CoT baseline for Qwen2.5-7B and Phi-4-mini.
- `fallback_order_sensitivity_fixed.csv`: Accuracy and parse-coverage results for alternative deterministic fallback orders.

The confidence calibration files are stored under `results/review_requested/`:

- `confidence_bin_calibration.csv`
- `confidence_summary_by_group.csv`
