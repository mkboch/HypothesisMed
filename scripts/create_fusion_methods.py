import json
from pathlib import Path
from collections import Counter
import math
import pandas as pd

def load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return {r["id"]: r for r in rows}, rows

cot_map, cot_rows = load("results/qwen2_5_7b_instruct_cot_medqa_original1000.jsonl")
direct_map, direct_rows = load("results/qwen2_5_7b_instruct_direct_medqa_original1000.jsonl")
v3_map, v3_rows = load("results/qwen2_5_7b_instruct_hypothesismed_v3_medqa_original1000.jsonl")

ids = [r["id"] for r in v3_rows]

def valid_letter(x):
    return isinstance(x, str) and x.strip().upper() in list("ABCDE")

def majority_vote(vals):
    vals = [v.strip().upper() for v in vals if valid_letter(v)]
    if not vals:
        return None
    c = Counter(vals)
    top_count = max(c.values())
    top = [k for k, v in c.items() if v == top_count]
    # Tie-breaking order: CoT, Direct, HypothesisMed-v3
    for preferred in vals:
        if preferred in top:
            return preferred
    return top[0]

def make_base_row(i):
    r = dict(v3_map[i])
    r["raw_output"] = ""
    r["parsed_output"] = {}
    return r

fusion_files = []

# Fusion 1: CoT answer, fallback to Direct, fallback to v3. SPACE and confidence from v3.
rows = []
for i in ids:
    r = make_base_row(i)
    ans = cot_map.get(i, {}).get("pred_answer")
    if not valid_letter(ans):
        ans = direct_map.get(i, {}).get("pred_answer")
    if not valid_letter(ans):
        ans = v3_map.get(i, {}).get("pred_answer")

    r["model"] = "qwen2_5_7b_instruct"
    r["method"] = "fusion_cot_direct_answer_hypmed_v3_space"
    r["pred_answer"] = ans.strip().upper() if valid_letter(ans) else None
    r["pred_space_label"] = v3_map.get(i, {}).get("pred_space_label")
    r["confidence"] = v3_map.get(i, {}).get("confidence", 0.0)
    rows.append(r)

out = Path("results/fusion/qwen2_5_7b_instruct_fusion_cot_direct_answer_hypmed_v3_space_medqa_original1000.jsonl")
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fusion_files.append(out)

# Fusion 2: Majority vote among CoT, Direct, v3. SPACE and confidence from v3.
rows = []
for i in ids:
    r = make_base_row(i)
    vals = [
        cot_map.get(i, {}).get("pred_answer"),
        direct_map.get(i, {}).get("pred_answer"),
        v3_map.get(i, {}).get("pred_answer"),
    ]
    ans = majority_vote(vals)

    r["model"] = "qwen2_5_7b_instruct"
    r["method"] = "fusion_majority_answer_hypmed_v3_space"
    r["pred_answer"] = ans
    r["pred_space_label"] = v3_map.get(i, {}).get("pred_space_label")
    r["confidence"] = v3_map.get(i, {}).get("confidence", 0.0)
    rows.append(r)

out = Path("results/fusion/qwen2_5_7b_instruct_fusion_majority_answer_hypmed_v3_space_medqa_original1000.jsonl")
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fusion_files.append(out)

# Fusion 3: CoT answer only, but v3 SPACE and confidence.
rows = []
for i in ids:
    r = make_base_row(i)
    ans = cot_map.get(i, {}).get("pred_answer")

    r["model"] = "qwen2_5_7b_instruct"
    r["method"] = "fusion_cot_answer_hypmed_v3_space"
    r["pred_answer"] = ans.strip().upper() if valid_letter(ans) else None
    r["pred_space_label"] = v3_map.get(i, {}).get("pred_space_label")
    r["confidence"] = v3_map.get(i, {}).get("confidence", 0.0)
    rows.append(r)

out = Path("results/fusion/qwen2_5_7b_instruct_fusion_cot_answer_hypmed_v3_space_medqa_original1000.jsonl")
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fusion_files.append(out)

def ci95(p, n):
    if n == 0:
        return ""
    se = math.sqrt(p * (1 - p) / n)
    return f"[{max(0,p-1.96*se):.3f}, {min(1,p+1.96*se):.3f}]"

def summarize(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    n = len(rows)
    acc = sum(r.get("pred_answer") == r.get("gold_answer") for r in rows) / n
    parse_cov = sum(r.get("pred_answer") is not None for r in rows) / n

    space_rows = [r for r in rows if r.get("pred_space_label") is not None]
    space_cov = len(space_rows) / n
    space_acc = (
        sum(r.get("pred_space_label") == r.get("gold_space_label") for r in space_rows) / len(space_rows)
        if space_rows else None
    )

    wrong = [
        r for r in rows
        if r.get("pred_answer") is not None and r.get("pred_answer") != r.get("gold_answer")
    ]
    high_wrong = [
        r for r in wrong
        if float(r.get("confidence") or 0.0) >= 0.8
    ]
    fcr = len(high_wrong) / len(wrong) if wrong else 0.0

    return {
        "file": str(path),
        "model": rows[0].get("model"),
        "method": rows[0].get("method"),
        "n": n,
        "answer_accuracy": round(acc, 4),
        "accuracy_ci95": ci95(acc, n),
        "parse_coverage": round(parse_cov, 4),
        "space_label_coverage": round(space_cov, 4),
        "space_label_accuracy": "" if space_acc is None else round(space_acc, 4),
        "false_commitment_wrong_cond": round(fcr, 4),
    }

original_files = [
    "results/qwen2_5_7b_instruct_cot_medqa_original1000.jsonl",
    "results/qwen2_5_7b_instruct_direct_medqa_original1000.jsonl",
    "results/qwen2_5_7b_instruct_hypothesismed_v2_medqa_original1000.jsonl",
    "results/qwen2_5_7b_instruct_hypothesismed_v3_medqa_original1000.jsonl",
]

all_files = [Path(x) for x in original_files if Path(x).exists()] + fusion_files
summary = pd.DataFrame([summarize(p) for p in all_files])
summary = summary.sort_values(["answer_accuracy", "parse_coverage"], ascending=False)

Path("results/paper_ready_fusion").mkdir(parents=True, exist_ok=True)
summary.to_csv("results/paper_ready_fusion/medqa_original1000_fusion_summary.csv", index=False)
summary.to_latex("results/paper_ready_fusion/medqa_original1000_fusion_summary.tex", index=False)

print(summary.to_string(index=False))
print("\nSaved:")
print("results/paper_ready_fusion/medqa_original1000_fusion_summary.csv")
print("results/paper_ready_fusion/medqa_original1000_fusion_summary.tex")
