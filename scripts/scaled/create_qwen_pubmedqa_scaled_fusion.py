#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

MODEL = "qwen2_5_7b_instruct"
DATASET_STEM = "pubmedqa_main_large"

ROOT = Path("/home/manikm/HypothesisMed")
RESULTS = ROOT / "results"
FUSION_DIR = RESULTS / "fusion"
FUSION_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    "direct": RESULTS / f"{MODEL}_direct_{DATASET_STEM}.jsonl",
    "cot": RESULTS / f"{MODEL}_cot_{DATASET_STEM}.jsonl",
    "hypothesismed_v3": RESULTS / f"{MODEL}_hypothesismed_v3_{DATASET_STEM}.jsonl",
}

out_path = FUSION_DIR / f"{MODEL}_fusion_majority_answer_hypmed_v3_space_{DATASET_STEM}.jsonl"

def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows

def choose_answer(direct, cot, hyp):
    candidates = []
    for name, row in [("direct", direct), ("cot", cot), ("hypothesismed_v3", hyp)]:
        ans = row.get("pred_answer")
        if ans:
            candidates.append((name, ans))

    counts = Counter(ans for _, ans in candidates)
    if counts:
        top_ans, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            return top_ans, "majority"

    # Deterministic fallback order used in paper.
    for name, row in [("direct", direct), ("cot", cot), ("hypothesismed_v3", hyp)]:
        ans = row.get("pred_answer")
        if ans:
            return ans, f"fallback_{name}"

    return None, "unparseable"

def main():
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k}: {p}")
        print(f"{k}: {sum(1 for _ in p.open())} rows | {p}")

    direct_rows = load_jsonl(paths["direct"])
    cot_rows = load_jsonl(paths["cot"])
    hyp_rows = load_jsonl(paths["hypothesismed_v3"])

    ids = sorted(set(direct_rows) & set(cot_rows) & set(hyp_rows))
    print(f"Shared ids: {len(ids)}")

    out_rows = []
    for id_ in ids:
        d = direct_rows[id_]
        c = cot_rows[id_]
        h = hyp_rows[id_]

        pred_answer, fusion_source = choose_answer(d, c, h)

        result = dict(h)
        result["method"] = "fusion_majority_answer_hypmed_v3_space"
        result["model"] = MODEL
        result["pred_answer"] = pred_answer
        result["fusion_source"] = fusion_source
        result["direct_answer"] = d.get("pred_answer")
        result["cot_answer"] = c.get("pred_answer")
        result["hypothesismed_v3_answer"] = h.get("pred_answer")

        # Reliability metadata comes from HypothesisMed-v3.
        result["pred_space_label"] = h.get("pred_space_label")
        result["confidence"] = h.get("confidence", 0.0)

        out_rows.append(result)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path}")

    n = len(out_rows)
    parsed = sum(1 for r in out_rows if r.get("pred_answer"))
    correct = sum(1 for r in out_rows if r.get("pred_answer") == r.get("gold_answer"))
    space_cov = sum(1 for r in out_rows if r.get("pred_space_label"))
    space_correct = sum(
        1 for r in out_rows
        if r.get("pred_space_label") and r.get("pred_space_label") == r.get("gold_space_label")
    )
    wrong = sum(
        1 for r in out_rows
        if r.get("pred_answer") and r.get("pred_answer") != r.get("gold_answer")
    )
    high_conf_wrong = sum(
        1 for r in out_rows
        if r.get("pred_answer")
        and r.get("pred_answer") != r.get("gold_answer")
        and float(r.get("confidence") or 0.0) >= 0.5
    )

    print()
    print("===== QWEN PUBMEDQA PROF-REVISION FUSION SUMMARY =====")
    print(f"n={n}")
    print(f"answer_accuracy={correct/n if n else 0:.4f}")
    print(f"parse_coverage={parsed/n if n else 0:.4f}")
    print(f"space_coverage={space_cov/n if n else 0:.4f}")
    print(f"space_accuracy_extracted={space_correct/space_cov if space_cov else 0:.4f}")
    print(f"false_commitment_wrong_cond={high_conf_wrong/wrong if wrong else 0:.4f}")

    print()
    print("===== FUSION SOURCE COUNTS =====")
    for k, v in Counter(r.get("fusion_source") for r in out_rows).most_common():
        print(f"{k},{v}")

if __name__ == "__main__":
    main()
