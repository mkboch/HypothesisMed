#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import re
import itertools
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("HYPOTHESISMED_ROOT", "/home/manikm/HypothesisMed")).resolve()
OUT = ROOT / "results" / "final_experimental_checks"
OUT.mkdir(parents=True, exist_ok=True)

LETTERS = "ABCDEFGH"
RNG = random.Random(20260701)

def read_jsonl(path):
    p = Path(path)
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

def read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False, default=str)
                rr[k] = v
            w.writerow(rr)

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def to_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def wilson_ci(p, n, z=1.959963984540054):
    if p is None or n is None or n <= 0:
        return ("", "")
    p = max(0.0, min(1.0, float(p)))
    denom = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return (round(max(0.0, center-half), 6), round(min(1.0, center+half), 6))

def norm_answer(x):
    if x is None or isinstance(x, bool):
        return ""
    if isinstance(x, (int, float)):
        n = int(x)
        if 0 <= n < len(LETTERS):
            return LETTERS[n]
        if 1 <= n <= len(LETTERS):
            return LETTERS[n-1]
    if isinstance(x, dict):
        for k in ["answer", "choice", "selected_answer", "final_answer", "pred_answer"]:
            if k in x:
                y = norm_answer(x[k])
                if y:
                    return y
        for v in x.values():
            y = norm_answer(v)
            if y:
                return y
        return ""
    if isinstance(x, list):
        for v in x:
            y = norm_answer(v)
            if y:
                return y
        return ""
    s = str(x).strip()
    if not s:
        return ""
    us = s.upper()
    if us in LETTERS:
        return us
    if re.fullmatch(r"[A-H](\|[A-H])+", us):
        return ""
    m = re.search(r'\b(?:ANSWER|CHOICE|OPTION|FINAL ANSWER|SELECTED ANSWER)\b[^A-H]{0,50}\b([A-H])\b', us, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\s*([A-H])[\.\)\:\-]\s+", us)
    if m:
        return m.group(1).upper()
    m = re.search(r'"(?:answer|choice|selected_answer|final_answer)"\s*:\s*"?(A|B|C|D|E|F|G|H)"?', s, flags=re.I)
    if m:
        return m.group(1).upper()
    if len(us) <= 25:
        m = re.search(r"\b([A-H])\b", us)
        if m:
            return m.group(1).upper()
    return ""

def norm_space(x):
    if x is None:
        return ""
    if isinstance(x, dict):
        for k in ["space_label", "space", "label", "answer_space", "space_status"]:
            if k in x:
                y = norm_space(x[k])
                if y:
                    return y
        for v in x.values():
            y = norm_space(v)
            if y:
                return y
        return ""
    s = str(x).upper()
    if "CONTRADICT" in s or "DUPLICATE" in s or "NONUNIQUE" in s or "NON-UNIQUE" in s:
        return "CONTRADICTED"
    if "INCOMPLETE" in s or "MISSING" in s or "INSUFFICIENT" in s:
        return "INCOMPLETE"
    if re.search(r"\bVALID\b", s):
        return "VALID"
    return ""

def get_first(row, keys):
    for k in keys:
        if k in row and row.get(k) not in [None, ""]:
            return row.get(k)
    return ""

def count_lines(path):
    p = Path(path)
    if not p.exists():
        return -1
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return -1

def find_file(name, roots):
    for r in roots:
        p = ROOT / r / name
        if p.exists():
            return p
    hits = list((ROOT / "results").glob(f"**/{name}"))
    return hits[0] if hits else None

def format_pct(x):
    if x == "" or x is None:
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

# ---------------------------------------------------------------------
# 1) File inventory and expected-count checks
# ---------------------------------------------------------------------
search_roots = [
    "results/expanded_fix_noapi",
    "results/expanded_fix_noapi/outputs",
    "results/expanded_fix_vllm",
    "results/expanded_fix_vllm/outputs",
    "results/expanded_final_gap",
    "results/expanded_final_gap/outputs",
    "results/expanded_claude_batch",
    "results/expanded_claude_tool_batch",
]

expected_files = [
    ("noapi_scaled_wide_base_outputs", "scaled_wide_base_outputs.csv", 20366),
    ("qwen3_14b_qa", "qwen3_14b_qa.jsonl", 9000),
    ("qwen3_30b_a3b_qa", "qwen3_30b_a3b_qa.jsonl", 9000),
    ("qwen3_14b_space_v4", "qwen3_14b_space_v4.jsonl", 540),
    ("qwen3_14b_self_consistency_k5", "qwen3_14b_self_consistency_k5.jsonl", 900),
    ("med42_qa", "med42_llama3_8b_qa.jsonl", 9000),
    ("med42_space_v4", "med42_llama3_8b_space_v4.jsonl", 540),
    ("openbiollm_qa", "openbiollm_llama3_8b_qa.jsonl", 9000),
    ("openbiollm_space_v4", "openbiollm_llama3_8b_space_v4.jsonl", 540),
    ("medgemma_1_5_4b_qa", "medgemma_1_5_4b_it_qa.jsonl", 9000),
    ("medgemma_1_5_4b_space_v4", "medgemma_1_5_4b_it_space_v4.jsonl", 540),
    ("medgemma_27b_text_qa", "medgemma_27b_text_it_qa.jsonl", 9000),
    ("medgemma_27b_text_space_v4", "medgemma_27b_text_it_space_v4.jsonl", 540),
    ("claude_tool_qa", "claude_haiku_4_5_tool_batch_qa.jsonl", 9000),
    ("claude_tool_space_v4", "claude_haiku_4_5_tool_batch_space_v4.jsonl", 540),
    ("claude_tool_raw_results", "claude_haiku_tool_batch_raw_results.jsonl", 9540),
]

inventory_rows = []
for label, fname, expected in expected_files:
    p = find_file(fname, search_roots)
    n = count_lines(p) if p else -1
    status = "OK" if p and n == expected else ("MISSING" if not p else "COUNT_MISMATCH")
    inventory_rows.append({
        "label": label,
        "file": fname,
        "path": str(p.relative_to(ROOT)) if p else "",
        "expected_rows": expected,
        "observed_rows": n if n >= 0 else "",
        "status": status,
    })

write_csv(OUT / "table_experiment_file_inventory.csv", inventory_rows)

# ---------------------------------------------------------------------
# 2) Unified confidence intervals for aggregate result tables
# ---------------------------------------------------------------------
metric_candidates = [
    "accuracy",
    "space_v4_accuracy",
    "hybrid_space_accuracy",
    "parse_coverage",
    "space_coverage",
    "accepted_accuracy",
    "coverage",
    "full_accuracy",
]

ci_rows = []
result_dir = ROOT / "results"
for p in sorted(result_dir.glob("**/table*.csv")):
    if "final_experimental_checks" in str(p):
        continue
    rows = read_csv(p)
    if not rows:
        continue
    for row in rows:
        n = to_int(row.get("N") or row.get("n") or row.get("count") or row.get("requests"))
        for m in metric_candidates:
            if m not in row:
                continue
            val = to_float(row.get(m))
            if val is None or val < 0 or val > 1 or not n:
                continue
            lo, hi = wilson_ci(val, n)
            ci_rows.append({
                "source_file": str(p.relative_to(ROOT)),
                "model": row.get("model", ""),
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "variant": row.get("variant", ""),
                "gold_space_norm": row.get("gold_space_norm", ""),
                "N": n,
                "metric": m,
                "value": round(val, 6),
                "ci95_low": lo,
                "ci95_high": hi,
            })

write_csv(OUT / "table_all_reportable_metrics_with_ci.csv", ci_rows)

best_rows = []
groups = {}
for r in ci_rows:
    if r["metric"] != "accuracy":
        continue
    model = r.get("model") or ""
    dataset = r.get("dataset") or ""
    if not model or not dataset:
        continue
    label = (r.get("method") or r.get("variant") or "").lower()
    if "oracle" in label:
        continue
    key = (model, dataset)
    if key not in groups or float(r["value"]) > float(groups[key]["value"]):
        groups[key] = r
for (model, dataset), r in sorted(groups.items()):
    best_rows.append({
        "model": model,
        "dataset": dataset,
        "best_setting": r.get("method") or r.get("variant"),
        "N": r["N"],
        "accuracy": r["value"],
        "ci95_low": r["ci95_low"],
        "ci95_high": r["ci95_high"],
        "source_file": r["source_file"],
    })
write_csv(OUT / "table_best_accuracy_by_model_dataset_with_ci.csv", best_rows)

# ---------------------------------------------------------------------
# 3) Claude tool-batch sanity audit
# ---------------------------------------------------------------------
claude_dir = ROOT / "results" / "expanded_claude_tool_batch"
claude_qa = read_jsonl(claude_dir / "claude_haiku_4_5_tool_batch_qa.jsonl")
claude_space = read_jsonl(claude_dir / "claude_haiku_4_5_tool_batch_space_v4.jsonl")
claude_meta = read_jsonl(claude_dir / "claude_haiku_tool_batch_metadata.jsonl")

claude_audit_rows = []

if claude_qa:
    g = defaultdict(list)
    for r in claude_qa:
        g[(r.get("dataset", ""), r.get("method", ""))].append(r)
    for (dataset, method), rows in sorted(g.items()):
        n = len(rows)
        pred_n = sum(1 for r in rows if norm_answer(r.get("pred_answer")))
        gold_n = sum(1 for r in rows if norm_answer(r.get("gold_answer")))
        correct_n = sum(1 for r in rows if str(r.get("correct")) in {"1", "1.0", "True", "true"})
        acc = correct_n / n if n else 0
        lo, hi = wilson_ci(acc, n)
        claude_audit_rows.append({
            "check": "claude_tool_qa_item_level",
            "dataset": dataset,
            "method": method,
            "N": n,
            "pred_coverage": round(pred_n / n, 6) if n else "",
            "gold_coverage": round(gold_n / n, 6) if n else "",
            "accuracy": round(acc, 6),
            "ci95_low": lo,
            "ci95_high": hi,
            "status": "OK" if pred_n == n and gold_n == n else "CHECK",
        })

if claude_space:
    n = len(claude_space)
    pred_n = sum(1 for r in claude_space if norm_space(r.get("pred_space")))
    gold_n = sum(1 for r in claude_space if norm_space(r.get("gold_space_norm")))
    correct_n = sum(1 for r in claude_space if str(r.get("space_correct")) in {"1", "1.0", "True", "true"})
    acc = correct_n / n if n else 0
    lo, hi = wilson_ci(acc, n)
    claude_audit_rows.append({
        "check": "claude_tool_space_item_level",
        "dataset": "ALL",
        "method": "space_v4",
        "N": n,
        "pred_coverage": round(pred_n / n, 6) if n else "",
        "gold_coverage": round(gold_n / n, 6) if n else "",
        "accuracy": round(acc, 6),
        "ci95_low": lo,
        "ci95_high": hi,
        "status": "OK" if pred_n == n and gold_n == n else "CHECK",
    })

leak_markers = [
    "gold_answer",
    "gold_space_norm",
    "correct_option",
    "correct_idx",
    "correct_index",
    "answer_idx",
    "answer_index",
    "label_idx",
]
leak_hits = []
for m in claude_meta:
    prompt = str(m.get("prompt", ""))
    low = prompt.lower()
    hits = [x for x in leak_markers if x.lower() in low]
    if hits:
        leak_hits.append({
            "custom_id": m.get("custom_id", ""),
            "task": m.get("task", ""),
            "dataset": m.get("dataset", ""),
            "method": m.get("method", ""),
            "markers": ",".join(hits),
        })

claude_audit_rows.append({
    "check": "claude_prompt_literal_leak_scan",
    "dataset": "ALL",
    "method": "ALL",
    "N": len(claude_meta),
    "pred_coverage": "",
    "gold_coverage": "",
    "accuracy": "",
    "ci95_low": "",
    "ci95_high": "",
    "status": "OK" if not leak_hits else f"CHECK_{len(leak_hits)}_HITS",
})

write_csv(
    OUT / "table_claude_tool_sanity_audit.csv",
    claude_audit_rows,
    ["check", "dataset", "method", "N", "pred_coverage", "gold_coverage", "accuracy", "ci95_low", "ci95_high", "status"],
)
write_jsonl(OUT / "claude_prompt_literal_leak_hits.jsonl", leak_hits)

# MedQA sanity sample
medqa_rows = [r for r in claude_qa if r.get("dataset") == "medqa"]
by_m = defaultdict(list)
for r in medqa_rows:
    by_m[r.get("method", "")].append(r)

sample = []
for method, rows in sorted(by_m.items()):
    rows = list(rows)
    RNG.shuffle(rows)
    sample.extend(rows[:17])
sample = sample[:50]

meta_by_id = {m.get("custom_id"): m for m in claude_meta}
sample_out = []
for r in sample:
    m = meta_by_id.get(r.get("custom_id"), {})
    prompt = m.get("prompt", "")
    q = ""
    opts = ""
    if "Question:" in prompt and "Options:" in prompt:
        q = prompt.split("Question:", 1)[1].split("Options:", 1)[0].strip()
        opts = prompt.split("Options:", 1)[1].strip()
    sample_out.append({
        "custom_id": r.get("custom_id"),
        "dataset": r.get("dataset"),
        "method": r.get("method"),
        "gold_answer": r.get("gold_answer"),
        "pred_answer": r.get("pred_answer"),
        "correct": r.get("correct"),
        "tool_input": r.get("tool_input"),
        "question_start": q[:500],
        "options_start": opts[:900],
    })
write_jsonl(OUT / "claude_medqa_tool_sanity_sample_50.jsonl", sample_out)

# ---------------------------------------------------------------------
# 4) Claude paired bootstrap for prompt/fusion differences
# ---------------------------------------------------------------------
def fusion_pick(preds, order):
    vals = [preds.get(k, "") for k in order if preds.get(k, "")]
    if not vals:
        return ""
    c = Counter(vals)
    top = max(c.values())
    winners = {k for k, v in c.items() if v == top}
    for k in order:
        if preds.get(k, "") in winners:
            return preds.get(k, "")
    return vals[0]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def quantile(xs, q):
    if not xs:
        return ""
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def bootstrap_diff(a, b, reps=2000):
    n = len(a)
    if n == 0:
        return ("", "", "")
    obs = mean(a) - mean(b)
    diffs = []
    for _ in range(reps):
        s = [RNG.randrange(n) for _ in range(n)]
        diffs.append(mean([a[i] for i in s]) - mean([b[i] for i in s]))
    return (round(obs, 6), round(quantile(diffs, 0.025), 6), round(quantile(diffs, 0.975), 6))

bootstrap_rows = []
if claude_qa:
    by_dataset_item = defaultdict(lambda: defaultdict(dict))
    gold_by_item = {}
    for r in claude_qa:
        d = r.get("dataset", "")
        item = r.get("item_id", "")
        method = r.get("method", "")
        pred = norm_answer(r.get("pred_answer"))
        gold = norm_answer(r.get("gold_answer"))
        if d and item and method:
            by_dataset_item[d][item][method] = pred
            if gold:
                gold_by_item[(d, item)] = gold

    for dataset, items in sorted(by_dataset_item.items()):
        item_ids = sorted(items.keys())
        methods = ["direct", "cot", "hypmed_v4"]
        acc_arrays = {}
        for method in methods:
            acc_arrays[method] = [
                1 if items[item].get(method, "") == gold_by_item.get((dataset, item), "") else 0
                for item in item_ids
            ]

        for variant, order in [
            ("majority_CDH", ["cot", "direct", "hypmed_v4"]),
            ("majority_DCH", ["direct", "cot", "hypmed_v4"]),
        ]:
            acc_arrays[variant] = [
                1 if fusion_pick(items[item], order) == gold_by_item.get((dataset, item), "") else 0
                for item in item_ids
            ]

        comparisons = [
            ("direct_minus_cot", "direct", "cot"),
            ("hypmed_v4_minus_direct", "hypmed_v4", "direct"),
            ("hypmed_v4_minus_cot", "hypmed_v4", "cot"),
            ("majority_CDH_minus_direct", "majority_CDH", "direct"),
            ("majority_CDH_minus_cot", "majority_CDH", "cot"),
            ("majority_DCH_minus_direct", "majority_DCH", "direct"),
            ("majority_DCH_minus_cot", "majority_DCH", "cot"),
        ]
        for label, a_name, b_name in comparisons:
            obs, lo, hi = bootstrap_diff(acc_arrays[a_name], acc_arrays[b_name])
            bootstrap_rows.append({
                "model": "claude_haiku_4_5_tool_batch",
                "dataset": dataset,
                "comparison": label,
                "N": len(item_ids),
                "accuracy_a": round(mean(acc_arrays[a_name]), 6),
                "accuracy_b": round(mean(acc_arrays[b_name]), 6),
                "diff_a_minus_b": obs,
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
            })

write_csv(OUT / "table_claude_tool_paired_bootstrap.csv", bootstrap_rows)

# ---------------------------------------------------------------------
# 5) OpenBioLLM diagnostic and label-permutation check
# ---------------------------------------------------------------------
openbio_path = find_file("openbiollm_llama3_8b_qa.jsonl", search_roots)
openbio_rows = read_jsonl(openbio_path) if openbio_path else []

def parse_from_raw(text):
    if text is None:
        return ""
    s = str(text)
    # Try JSON first.
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = norm_answer(obj)
            if ans:
                return ans
        except Exception:
            pass
    return norm_answer(s)

def extract_raw(row):
    for k in ["raw_text", "text", "response", "output", "completion", "model_output", "raw", "answer_text", "generated_text"]:
        if k in row and row.get(k) not in [None, ""]:
            return row.get(k)
    # nested common fields
    for k, v in row.items():
        if isinstance(v, dict):
            got = extract_raw(v)
            if got:
                return got
    return ""

openbio_diag = []
openbio_examples = []
openbio_reparsed_rows = []

if openbio_rows:
    groups2 = defaultdict(list)
    for r in openbio_rows:
        dataset = get_first(r, ["dataset", "data", "benchmark"]) or "unknown"
        method = get_first(r, ["method", "mode", "prompt_mode", "variant"]) or "unknown"
        groups2[(dataset, method)].append(r)

    for (dataset, method), rows in sorted(groups2.items()):
        orig_preds = []
        reparsed_preds = []
        golds = []
        raw_examples = []

        for r in rows:
            gold = norm_answer(get_first(r, ["gold_answer", "gold", "label", "target", "correct_answer", "correct_option"]))
            orig = norm_answer(get_first(r, ["pred_answer", "prediction", "pred", "selected_answer", "model_answer", "parsed_answer"]))
            raw = extract_raw(r)
            rep = orig or parse_from_raw(raw)

            golds.append(gold)
            orig_preds.append(orig)
            reparsed_preds.append(rep)

            rr = dict(r)
            rr["_openbio_reparsed_answer"] = rep
            rr["_openbio_gold_answer_norm"] = gold
            rr["_openbio_reparsed_correct"] = int(rep == gold) if rep and gold else 0
            openbio_reparsed_rows.append(rr)

            if len(raw_examples) < 5:
                raw_examples.append({
                    "dataset": dataset,
                    "method": method,
                    "gold": gold,
                    "orig_pred": orig,
                    "reparsed_pred": rep,
                    "raw_text_start": str(raw)[:1500],
                    "keys": sorted(list(r.keys()))[:80],
                })

        n = len(rows)
        orig_cov = sum(1 for x in orig_preds if x) / n if n else 0
        rep_cov = sum(1 for x in reparsed_preds if x) / n if n else 0
        orig_acc = sum(1 for p, g in zip(orig_preds, golds) if p and g and p == g) / n if n else 0
        rep_acc = sum(1 for p, g in zip(reparsed_preds, golds) if p and g and p == g) / n if n else 0

        letters = sorted(set([x for x in reparsed_preds + golds if x and x in "ABCDE"]))
        best_perm_acc = rep_acc
        best_perm_map = {x: x for x in letters}
        if 2 <= len(letters) <= 5:
            for perm in itertools.permutations(letters):
                mp = dict(zip(letters, perm))
                acc = sum(1 for p, g in zip(reparsed_preds, golds) if p and g and mp.get(p, p) == g) / n if n else 0
                if acc > best_perm_acc:
                    best_perm_acc = acc
                    best_perm_map = mp

        pred_dist = Counter([x or "EMPTY" for x in reparsed_preds])
        gold_dist = Counter([x or "EMPTY" for x in golds])

        openbio_diag.append({
            "model": "openbiollm_llama3_8b",
            "dataset": dataset,
            "method": method,
            "N": n,
            "orig_parse_coverage": round(orig_cov, 6),
            "orig_accuracy": round(orig_acc, 6),
            "reparse_coverage": round(rep_cov, 6),
            "reparse_accuracy": round(rep_acc, 6),
            "best_label_permutation_accuracy": round(best_perm_acc, 6),
            "best_label_permutation_map": json.dumps(best_perm_map, sort_keys=True),
            "reparsed_pred_distribution": json.dumps(dict(pred_dist), sort_keys=True),
            "gold_distribution": json.dumps(dict(gold_dist), sort_keys=True),
            "recommendation": "mapping_issue_possible" if best_perm_acc >= rep_acc + 0.20 else ("appendix_or_omit" if rep_acc < 0.20 else "usable_with_caution"),
        })
        openbio_examples.extend(raw_examples)

write_csv(OUT / "table_openbiollm_reparse_diagnostic.csv", openbio_diag)
write_jsonl(OUT / "openbiollm_raw_examples_for_diagnostic.jsonl", openbio_examples)
write_jsonl(OUT / "openbiollm_llama3_8b_qa_reparsed_diagnostic.jsonl", openbio_reparsed_rows)

# ---------------------------------------------------------------------
# 6) Final gap report
# ---------------------------------------------------------------------
inventory_bad = [r for r in inventory_rows if r["status"] != "OK"]
claude_tool_rows = read_csv(claude_dir / "table_claude_haiku_tool_by_method.csv")
claude_tool_space = read_csv(claude_dir / "table_claude_haiku_tool_space_v4_overall.csv")
claude_tool_cost = read_csv(claude_dir / "table_claude_haiku_tool_actual_usage_cost.csv")

def rows_to_md(rows, fields, max_rows=30):
    if not rows:
        return "_No rows._\n"
    rows = rows[:max_rows]
    out = []
    out.append("| " + " | ".join(fields) + " |")
    out.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    return "\n".join(out) + "\n"

summary = []
summary.append("# Final Experimental Checks\n")
summary.append("## Status\n")
summary.append("No new model calls or API calls were made by this script. It only analyzed saved outputs.\n")
summary.append(f"Project root: `{ROOT}`\n")
summary.append("\n## Expected-file inventory\n")
summary.append(rows_to_md(inventory_rows, ["label", "expected_rows", "observed_rows", "status"], max_rows=50))

summary.append("\n## Claude tool-batch sanity audit\n")
summary.append(rows_to_md(claude_audit_rows, ["check", "dataset", "method", "N", "pred_coverage", "gold_coverage", "accuracy", "ci95_low", "ci95_high", "status"], max_rows=50))

summary.append("\n## Claude tool-batch QA table\n")
summary.append(rows_to_md(claude_tool_rows, ["model", "dataset", "method", "N", "accuracy", "parse_coverage", "space_coverage"], max_rows=30))

summary.append("\n## Claude tool-batch SPACE overall\n")
summary.append(rows_to_md(claude_tool_space, ["model", "N", "space_coverage", "space_v4_accuracy", "hybrid_space_accuracy"], max_rows=10))

summary.append("\n## Claude tool-batch cost\n")
summary.append(rows_to_md(claude_tool_cost, ["model", "requests", "input_tokens", "output_tokens", "actual_batch_cost_usd"], max_rows=10))

summary.append("\n## Claude paired bootstrap comparisons\n")
summary.append(rows_to_md(bootstrap_rows, ["dataset", "comparison", "N", "accuracy_a", "accuracy_b", "diff_a_minus_b", "bootstrap95_low", "bootstrap95_high"], max_rows=60))

summary.append("\n## OpenBioLLM diagnostic\n")
summary.append(rows_to_md(openbio_diag, ["dataset", "method", "N", "orig_accuracy", "reparse_accuracy", "best_label_permutation_accuracy", "recommendation"], max_rows=30))

summary.append("\n## Final experimental gap assessment\n")
if inventory_bad:
    summary.append(f"- Some expected artifacts are missing or count-mismatched: {len(inventory_bad)}. See `table_experiment_file_inventory.csv`.\n")
else:
    summary.append("- Expected major artifacts are present with expected counts.\n")

summary.append("- Remaining work before manuscript patching is analysis/reporting, not new model experiments.\n")
summary.append("- Confidence intervals were generated for reportable aggregate metrics.\n")
summary.append("- Claude tool-batch item-level sanity checks and bootstrap comparisons were generated.\n")
summary.append("- OpenBioLLM was diagnostically reparsed and checked for possible label-permutation issues.\n")
summary.append("- The earlier Claude free-text batch should remain diagnostic only. The Claude tool-batch output is the clean API result.\n")

summary.append("\n## Generated files\n")
for p in sorted(OUT.glob("*")):
    summary.append(f"- `{p.relative_to(ROOT)}`\n")

summary_text = "\n".join(summary)
(OUT / "FINAL_EXPERIMENTAL_CHECKS_SUMMARY.md").write_text(summary_text, encoding="utf-8")

print(summary_text)
print("\nFINAL_EXPERIMENTAL_CHECKS_OK")
