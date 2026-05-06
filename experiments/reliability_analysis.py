import argparse
import json
import math
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss

def load(path):
    return pd.read_json(path, lines=True)

def ece(df, bins=10):
    d = df.copy()
    d["correct"] = (d["pred_answer"] == d["gold_answer"]).astype(float)
    d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce").fillna(0.0).clip(0, 1)
    out = 0.0
    n = len(d)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        b = d[(d["confidence"] >= lo) & (d["confidence"] < hi if i < bins - 1 else d["confidence"] <= hi)]
        if len(b) == 0:
            continue
        out += len(b) / n * abs(b["correct"].mean() - b["confidence"].mean())
    return out

def brier(df):
    y = (df["pred_answer"] == df["gold_answer"]).astype(int)
    c = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).clip(0, 1)
    return brier_score_loss(y, c)

def bootstrap_ci(df, metric_fn, rounds=1000, seed=42):
    rng = random.Random(seed)
    vals = []
    rows = df.to_dict("records")
    n = len(rows)
    for _ in range(rounds):
        sample = pd.DataFrame([rows[rng.randrange(n)] for _ in range(n)])
        vals.append(metric_fn(sample))
    vals = sorted(vals)
    return vals[int(0.025 * rounds)], vals[int(0.975 * rounds)]

def acc(df):
    return float((df["pred_answer"] == df["gold_answer"]).mean())

def fcr(df):
    wrong = df[df["pred_answer"] != df["gold_answer"]]
    if len(wrong) == 0:
        return 0.0
    return float((wrong["pred_space_label"] == "VALID").mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    rows = []
    for path in args.paths:
        name = Path(path).stem
        df = load(path)
        metrics = {
            "run": name,
            "n": len(df),
            "accuracy": acc(df),
            "false_commitment": fcr(df),
            "ece": ece(df, args.bins),
            "brier": brier(df),
        }
        lo, hi = bootstrap_ci(df, acc)
        metrics["accuracy_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        lo, hi = bootstrap_ci(df, fcr)
        metrics["fcr_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(metrics)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    Path("results/summary").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/summary/reliability_summary.csv", index=False)
    print("\nSaved results/summary/reliability_summary.csv")

if __name__ == "__main__":
    main()
