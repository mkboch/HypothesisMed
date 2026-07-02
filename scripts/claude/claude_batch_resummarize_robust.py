#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/manikm/HypothesisMed")
OUT = ROOT / "results" / "expanded_claude_batch"
META_PATH = OUT / "claude_haiku_batch_metadata.jsonl"
RAW_RESULTS_PATH = OUT / "claude_haiku_batch_raw_results.jsonl"
MODEL_LABEL = "claude_haiku_4_5_batch"
LETTERS = "ABCDEFGH"

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
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
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def clean_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"```(?:json)?", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_options_from_prompt(prompt):
    opts = {}
    after = str(prompt).split("Options:", 1)
    if len(after) < 2:
        return opts
    text = after[1]
    for line in text.splitlines():
        m = re.match(r"^\s*([A-H])\s*[\.\)\:\-]\s*(.+?)\s*$", line)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
    return opts

def extract_message_text(result_obj):
    result = result_obj.get("result", {})
    if result.get("type") != "succeeded":
        return ""
    msg = result.get("message", {})
    content = msg.get("content", [])
    parts = []
    for c in content:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(str(c.get("text", "")))
    return "\n".join(parts).strip()

def extract_usage(result_obj):
    result = result_obj.get("result", {})
    if result.get("type") != "succeeded":
        return {"input_tokens": 0, "output_tokens": 0}
    usage = result.get("message", {}).get("usage", {}) or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }

def parse_json_object(text):
    if not text:
        return {}
    t = str(text).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Try balanced JSON-like substrings.
    starts = [m.start() for m in re.finditer(r"\{", t)]
    for st in starts:
        depth = 0
        for i in range(st, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = t[st:i+1]
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else {}
                    except Exception:
                        break
    return {}

def flatten_json_values(obj, parent_key=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = str(k)
            out.append((kk, v))
            out.extend(flatten_json_values(v, kk))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_json_values(v, parent_key))
    return out

def norm_answer_value(value, options):
    if value is None or isinstance(value, bool):
        return ""

    if isinstance(value, dict):
        for k, v in flatten_json_values(value):
            ans = norm_answer_value(v, options)
            if ans:
                return ans
        return ""

    if isinstance(value, list):
        for v in value:
            ans = norm_answer_value(v, options)
            if ans:
                return ans
        return ""

    if isinstance(value, (int, float)):
        n = int(value)
        if 0 <= n < len(LETTERS):
            return LETTERS[n]
        if 1 <= n <= len(LETTERS):
            return LETTERS[n - 1]

    s = str(value).strip()
    if not s:
        return ""

    us = s.upper().strip()

    # Ignore instruction echo.
    if re.fullmatch(r"[A-H](\|[A-H])+", us):
        return ""

    if us in LETTERS:
        return us

    m = re.match(r"^\s*(?:OPTION|CHOICE|ANSWER)?\s*([A-H])\s*[\.\)\:\-]\s+", us, flags=re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(?:ANSWER|CHOICE|OPTION|FINAL ANSWER|SELECTED ANSWER)\b[^A-H]{0,40}\b([A-H])\b", us, flags=re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r"\(([A-H])\)", us)
    if m and len(us) <= 200:
        return m.group(1).upper()

    if us.isdigit():
        n = int(us)
        if 0 <= n < len(LETTERS):
            return LETTERS[n]
        if 1 <= n <= len(LETTERS):
            return LETTERS[n - 1]

    cs = clean_text(s)
    opt_clean = {k: clean_text(v) for k, v in options.items()}

    for k, v in opt_clean.items():
        if cs == v and v:
            return k

    # Safe substring match for long option texts.
    for k, v in opt_clean.items():
        if v and len(v) >= 12 and (v in cs or cs in v):
            return k

    return ""

def parse_answer_robust(text, prompt):
    options = parse_options_from_prompt(prompt)
    obj = parse_json_object(text)

    preferred_keys = [
        "answer", "selected_answer", "selected_option", "choice", "option",
        "final_answer", "best_answer", "predicted_answer", "letter", "response"
    ]

    if obj:
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        for k in preferred_keys:
            if k in lower_map:
                ans = norm_answer_value(lower_map[k], options)
                if ans:
                    return ans

        for k, v in flatten_json_values(obj):
            if str(k).lower() in preferred_keys or "answer" in str(k).lower() or "choice" in str(k).lower() or "option" in str(k).lower():
                ans = norm_answer_value(v, options)
                if ans:
                    return ans

    ans = norm_answer_value(text, options)
    if ans:
        return ans

    # Last cautious fallback: one answer-like letter mention.
    t = str(text)
    pats = [
        r"\bfinal answer\s*(?:is|:)?\s*([A-H])\b",
        r"\banswer\s*(?:is|:)?\s*([A-H])\b",
        r"\bchoice\s*(?:is|:)?\s*([A-H])\b",
        r"\boption\s*([A-H])\b",
    ]
    for pat in pats:
        m = re.search(pat, t, flags=re.I)
        if m:
            return m.group(1).upper()

    return ""

def norm_space_label(x):
    if x is None:
        return ""
    s = str(x).upper()
    if "CONTRADICT" in s or "DUPLICATE" in s or "NONUNIQUE" in s or "NON-UNIQUE" in s:
        return "CONTRADICTED"
    if "INCOMPLETE" in s or "MISSING" in s or "INSUFFICIENT" in s:
        return "INCOMPLETE"
    if re.search(r"\bVALID\b", s):
        return "VALID"
    return ""

def parse_space_robust(text):
    obj = parse_json_object(text)
    if obj:
        preferred = ["space_label", "space", "label", "answer_space", "space_status", "validity"]
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        for k in preferred:
            if k in lower_map:
                y = norm_space_label(lower_map[k])
                if y:
                    return y
        for k, v in flatten_json_values(obj):
            if "space" in str(k).lower() or "label" in str(k).lower() or "valid" in str(k).lower():
                y = norm_space_label(v)
                if y:
                    return y
    return norm_space_label(text)

def safe_acc(vals):
    vals = [v for v in vals if v is not None and v != ""]
    if not vals:
        return ""
    return round(sum(vals) / len(vals), 6)

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

def main():
    if not META_PATH.exists():
        raise SystemExit(f"MISSING {META_PATH}")
    if not RAW_RESULTS_PATH.exists():
        raise SystemExit(f"MISSING {RAW_RESULTS_PATH}")

    meta = {r["custom_id"]: r for r in read_jsonl(META_PATH)}
    raw = read_jsonl(RAW_RESULTS_PATH)

    qa_rows = []
    space_rows = []
    status_counts = Counter()
    total_in = 0
    total_out = 0
    unparsed_examples = defaultdict(list)

    for obj in raw:
        cid = obj.get("custom_id")
        m = meta.get(cid, {})
        if not m:
            continue

        result_type = obj.get("result", {}).get("type", "unknown")
        status_counts[result_type] += 1
        text = extract_message_text(obj)
        usage = extract_usage(obj)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]

        base = {
            **m,
            "result_type": result_type,
            "raw_text": text,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

        if m.get("task") == "qa":
            pred = parse_answer_robust(text, m.get("prompt", ""))
            sp = parse_space_robust(text)
            gold = m.get("gold_answer", "")
            correct = int(pred == gold) if gold and pred else (0 if gold else None)
            row = {**base, "pred_answer": pred, "pred_space": sp, "correct": correct}
            qa_rows.append(row)
            if not pred and len(unparsed_examples[(m.get("dataset"), m.get("method"))]) < 5:
                unparsed_examples[(m.get("dataset"), m.get("method"))].append({
                    "custom_id": cid,
                    "gold_answer": gold,
                    "raw_text": text[:1500],
                    "prompt_tail": m.get("prompt", "")[-1200:],
                })

        elif m.get("task") == "space":
            sp = parse_space_robust(text)
            gold_sp = m.get("gold_space_norm", "")
            correct = int(sp == gold_sp) if gold_sp and sp else (0 if gold_sp else None)
            space_rows.append({**base, "pred_space": sp, "space_correct": correct})

    write_jsonl(OUT / "claude_haiku_4_5_batch_qa_robust.jsonl", qa_rows)
    write_jsonl(OUT / "claude_haiku_4_5_batch_space_v4_robust.jsonl", space_rows)

    by_method = []
    g = defaultdict(list)
    for r in qa_rows:
        g[(r["model"], r["dataset"], r["method"])].append(r)

    for (model, dataset, method), rows in sorted(g.items()):
        N = len(rows)
        by_method.append({
            "model": model,
            "dataset": dataset,
            "method": method,
            "N": N,
            "accuracy": safe_acc([r["correct"] for r in rows]),
            "parse_coverage": round(sum(1 for r in rows if r.get("pred_answer")) / N, 6) if N else "",
            "space_coverage": round(sum(1 for r in rows if r.get("pred_space")) / N, 6) if N else "",
        })

    write_csv(
        OUT / "table_claude_haiku_by_method_robust.csv",
        by_method,
        ["model", "dataset", "method", "N", "accuracy", "parse_coverage", "space_coverage"],
    )

    by_item = defaultdict(dict)
    gold_by_item = {}
    for r in qa_rows:
        key = (r["dataset"], r["item_id"])
        by_item[key][r["method"]] = r.get("pred_answer", "")
        if r.get("gold_answer"):
            gold_by_item[key] = r.get("gold_answer")

    fusion_rows = []
    for variant, order in [
        ("majority_CDH", ["cot", "direct", "hypmed_v4"]),
        ("majority_DCH", ["direct", "cot", "hypmed_v4"]),
    ]:
        by_dataset = defaultdict(list)
        for key, preds in by_item.items():
            dataset, item = key
            pred = fusion_pick(preds, order)
            gold = gold_by_item.get(key, "")
            correct = int(pred == gold) if gold and pred else (0 if gold else None)
            by_dataset[dataset].append({"pred": pred, "gold": gold, "correct": correct})
        for dataset, rows in sorted(by_dataset.items()):
            N = len(rows)
            fusion_rows.append({
                "model": MODEL_LABEL,
                "dataset": dataset,
                "variant": variant,
                "N": N,
                "accuracy": safe_acc([r["correct"] for r in rows]),
                "parse_coverage": round(sum(1 for r in rows if r.get("pred")) / N, 6) if N else "",
            })

    write_csv(
        OUT / "table_claude_haiku_fusion_robust.csv",
        fusion_rows,
        ["model", "dataset", "variant", "N", "accuracy", "parse_coverage"],
    )

    space_overall = []
    if space_rows:
        N = len(space_rows)
        space_overall.append({
            "model": MODEL_LABEL,
            "N": N,
            "space_coverage": round(sum(1 for r in space_rows if r.get("pred_space")) / N, 6),
            "space_v4_accuracy": safe_acc([r["space_correct"] for r in space_rows]),
            "hybrid_space_accuracy": safe_acc([r["space_correct"] for r in space_rows]),
        })

    write_csv(
        OUT / "table_claude_haiku_space_v4_overall_robust.csv",
        space_overall,
        ["model", "N", "space_coverage", "space_v4_accuracy", "hybrid_space_accuracy"],
    )

    by_label = defaultdict(list)
    for r in space_rows:
        by_label[r.get("gold_space_norm", "")].append(r)

    space_label_rows = []
    for label, rows in sorted(by_label.items()):
        if not label:
            continue
        N = len(rows)
        space_label_rows.append({
            "model": MODEL_LABEL,
            "gold_space_norm": label,
            "N": N,
            "space_coverage": round(sum(1 for r in rows if r.get("pred_space")) / N, 6),
            "space_v4_accuracy": safe_acc([r["space_correct"] for r in rows]),
            "hybrid_space_accuracy": safe_acc([r["space_correct"] for r in rows]),
        })

    write_csv(
        OUT / "table_claude_haiku_space_v4_by_label_robust.csv",
        space_label_rows,
        ["model", "gold_space_norm", "N", "space_coverage", "space_v4_accuracy", "hybrid_space_accuracy"],
    )

    actual_cost = total_in / 1_000_000 * 0.50 + total_out / 1_000_000 * 2.50
    usage_rows = [{
        "model": MODEL_LABEL,
        "requests": len(qa_rows) + len(space_rows),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "batch_input_usd_per_mtok": 0.50,
        "batch_output_usd_per_mtok": 2.50,
        "actual_batch_cost_usd": round(actual_cost, 6),
    }]
    write_csv(
        OUT / "table_claude_haiku_actual_usage_cost_robust.csv",
        usage_rows,
        ["model", "requests", "input_tokens", "output_tokens", "batch_input_usd_per_mtok", "batch_output_usd_per_mtok", "actual_batch_cost_usd"],
    )

    diag_path = OUT / "claude_haiku_unparsed_examples_after_robust.txt"
    with diag_path.open("w", encoding="utf-8") as f:
        for key, exs in sorted(unparsed_examples.items()):
            f.write(f"\n\n===== {key} =====\n")
            for ex in exs:
                f.write(json.dumps(ex, indent=2, ensure_ascii=False) + "\n")

    md = OUT / "CLAUDE_HAIKU_BATCH_SUMMARY_ROBUST.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Claude Haiku 4.5 Batch Results, Robust Reparse\n\n")
        f.write(f"Result status counts: `{dict(status_counts)}`\n\n")
        f.write(f"QA rows: **{len(qa_rows)}**\n\n")
        f.write(f"SPACE rows: **{len(space_rows)}**\n\n")
        f.write(f"Actual usage: input tokens **{total_in:,}**, output tokens **{total_out:,}**\n\n")
        f.write(f"Estimated actual batch cost: **${actual_cost:.4f}**\n\n")
        f.write("## QA by method\n\n")
        for r in by_method:
            f.write(json.dumps(r, ensure_ascii=False) + "\n\n")
        f.write("## Fusion\n\n")
        for r in fusion_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n\n")
        f.write("## SPACE overall\n\n")
        for r in space_overall:
            f.write(json.dumps(r, ensure_ascii=False) + "\n\n")
        f.write("## SPACE by label\n\n")
        for r in space_label_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n\n")

    print("=== ROBUST QA BY METHOD ===")
    for r in by_method:
        print(r)

    print("\n=== ROBUST FUSION ===")
    for r in fusion_rows:
        print(r)

    print("\n=== ROBUST SPACE OVERALL ===")
    for r in space_overall:
        print(r)

    print("\n=== ROBUST SPACE BY LABEL ===")
    for r in space_label_rows:
        print(r)

    print("\n=== FILES WRITTEN ===")
    for p in [
        OUT / "claude_haiku_4_5_batch_qa_robust.jsonl",
        OUT / "claude_haiku_4_5_batch_space_v4_robust.jsonl",
        OUT / "table_claude_haiku_by_method_robust.csv",
        OUT / "table_claude_haiku_fusion_robust.csv",
        OUT / "table_claude_haiku_space_v4_overall_robust.csv",
        OUT / "table_claude_haiku_space_v4_by_label_robust.csv",
        OUT / "table_claude_haiku_actual_usage_cost_robust.csv",
        md,
        diag_path,
    ]:
        print(p)

    print("\nCLAUDE_ROBUST_RESUMMARY_OK")

if __name__ == "__main__":
    main()
