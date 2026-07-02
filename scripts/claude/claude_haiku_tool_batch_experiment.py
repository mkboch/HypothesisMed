#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

ROOT = Path("/home/manikm/HypothesisMed")
PHASE2 = ROOT / "results" / "expanded_fix_vllm"
OUT = ROOT / "results" / "expanded_claude_tool_batch"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "claude-haiku-4-5-20251001"
MODEL_LABEL = "claude_haiku_4_5_tool_batch"
LETTERS = "ABCDEFGH"

STATE_PATH = OUT / "claude_haiku_tool_batch_state.json"
META_PATH = OUT / "claude_haiku_tool_batch_metadata.jsonl"
REQUESTS_PATH = OUT / "claude_haiku_tool_batch_requests.jsonl"
RAW_RESULTS_PATH = OUT / "claude_haiku_tool_batch_raw_results.jsonl"

QA_TOOL = {
    "name": "record_qa_answer",
    "description": "Record the selected answer for a biomedical multiple-choice question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": list("ABCDEFGH")},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "space_label": {"type": "string", "enum": ["VALID", "INCOMPLETE", "CONTRADICTED"]},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    },
}

SPACE_TOOL = {
    "name": "record_space_label",
    "description": "Record the SPACE validity label for a biomedical multiple-choice question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "space_label": {"type": "string", "enum": ["VALID", "INCOMPLETE", "CONTRADICTED"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["space_label", "confidence"],
        "additionalProperties": False,
    },
}

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

def parse_options(x):
    if isinstance(x, dict):
        return {str(k).strip().upper()[:1]: str(v) for k, v in x.items() if str(k).strip().upper()[:1] in LETTERS}
    if isinstance(x, list):
        return {LETTERS[i]: str(v) for i, v in enumerate(x[:len(LETTERS)])}
    try:
        return parse_options(json.loads(str(x)))
    except Exception:
        return {}

def fmt_options(x):
    opts = parse_options(x)
    if not opts:
        return str(x)
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))

def norm_answer(value, options=None):
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        n = int(value)
        if 0 <= n < len(LETTERS):
            return LETTERS[n]
        if 1 <= n <= len(LETTERS):
            return LETTERS[n - 1]
    s = str(value).strip().upper()
    if s in LETTERS:
        return s
    m = re.search(r"\b([A-H])\b", s)
    if m and len(s) <= 50:
        return m.group(1)
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

def extract_gold_answer(row):
    for k in ["gold_answer", "answer", "gold", "label", "target", "correct", "correct_answer", "correct_option", "answer_idx", "answer_index", "correct_idx", "correct_index", "label_idx"]:
        if k in row:
            ans = norm_answer(row.get(k), row.get("options", {}))
            if ans:
                return ans
    return ""

def extract_gold_space(row):
    for k in ["gold_space_norm", "gold_space", "space_gold", "target_space", "space_label_gold", "label"]:
        if k in row:
            y = norm_space_label(row.get(k))
            if y:
                return y
    return ""

def row_id(row, idx):
    for k in ["id", "question_id", "uid", "qid", "sample_id"]:
        if row.get(k) is not None:
            return str(row.get(k))
    return f"idx_{idx}"

def prompt_qa(row, mode):
    base = f"""Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""
    if mode == "direct":
        return "Select the best answer. Use the record_qa_answer tool only.\n\n" + base
    if mode == "cot":
        return "Think silently and select the best answer. Use the record_qa_answer tool only. Do not write reasoning.\n\n" + base
    if mode == "hypmed_v4":
        return """Evaluate the answer space, then select the best answer if possible.

SPACE labels:
VALID: exactly one medically supported best answer is present.
INCOMPLETE: the medically correct answer is missing or insufficiently represented.
CONTRADICTED: options are duplicated, non-unique, mutually inconsistent, or prevent one best answer.

Use the record_qa_answer tool only. Include space_label when possible.

""" + base
    raise ValueError(mode)

def prompt_space(row):
    return f"""Audit the answer-space validity of this biomedical multiple-choice question.

SPACE labels:
VALID: exactly one medically supported best answer is present.
INCOMPLETE: the medically correct answer is missing or insufficiently represented.
CONTRADICTED: options are duplicated, non-unique, mutually inconsistent, or prevent one best answer.

Use the record_space_label tool only.

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def to_plain(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if hasattr(x, "dict"):
        return x.dict()
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"raw": str(x)}

def build_payload(qa_max_per_dataset, space_max_rows, modes):
    qa_rows = read_jsonl(PHASE2 / "stronger_model_eval_input.jsonl")
    space_rows = read_jsonl(PHASE2 / "space_v4_stress_input.jsonl")

    requests = []
    metadata = []
    seq = 0

    by_dataset = defaultdict(int)
    kept_qa = []
    for i, row in enumerate(qa_rows):
        d = str(row.get("dataset", "unknown"))
        if qa_max_per_dataset > 0 and by_dataset[d] >= qa_max_per_dataset:
            continue
        by_dataset[d] += 1
        kept_qa.append((i, row))

    for original_index, row in kept_qa:
        dataset = str(row.get("dataset", "unknown"))
        rid = row_id(row, original_index)
        gold_answer = extract_gold_answer(row)

        for mode in modes:
            seq += 1
            cid = f"r{seq:07d}"
            req = {
                "custom_id": cid,
                "params": {
                    "model": MODEL,
                    "max_tokens": 96,
                    "temperature": 0,
                    "system": "You must use the provided tool. Do not write free text.",
                    "tools": [QA_TOOL],
                    "tool_choice": {"type": "tool", "name": "record_qa_answer"},
                    "messages": [{"role": "user", "content": prompt_qa(row, mode)}],
                },
            }
            meta = {
                "custom_id": cid,
                "task": "qa",
                "model": MODEL_LABEL,
                "dataset": dataset,
                "method": mode,
                "item_id": f"{dataset}:{rid}",
                "row_id": rid,
                "original_index": original_index,
                "gold_answer": gold_answer,
                "gold_space_norm": "",
            }
            requests.append(req)
            metadata.append(meta)

    if space_max_rows > 0:
        space_rows = space_rows[:space_max_rows]

    for i, row in enumerate(space_rows):
        seq += 1
        cid = f"r{seq:07d}"
        dataset = str(row.get("dataset", "stress"))
        rid = row_id(row, i)
        req = {
            "custom_id": cid,
            "params": {
                "model": MODEL,
                "max_tokens": 64,
                "temperature": 0,
                "system": "You must use the provided tool. Do not write free text.",
                "tools": [SPACE_TOOL],
                "tool_choice": {"type": "tool", "name": "record_space_label"},
                "messages": [{"role": "user", "content": prompt_space(row)}],
            },
        }
        meta = {
            "custom_id": cid,
            "task": "space",
            "model": MODEL_LABEL,
            "dataset": dataset,
            "method": "space_v4",
            "item_id": f"{dataset}:{rid}",
            "row_id": rid,
            "original_index": i,
            "gold_answer": "",
            "gold_space_norm": extract_gold_space(row),
        }
        requests.append(req)
        metadata.append(meta)

    return requests, metadata

def submit(args):
    if STATE_PATH.exists() and not args.force_new:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        print(f"EXISTING_STATE batch_id={state.get('id')} status={state.get('processing_status')}")
        return state.get("id")

    requests, metadata = build_payload(
        qa_max_per_dataset=args.qa_max_per_dataset,
        space_max_rows=args.space_max_rows,
        modes=[m.strip() for m in args.modes.split(",") if m.strip()],
    )
    write_jsonl(REQUESTS_PATH, requests)
    write_jsonl(META_PATH, metadata)

    print(f"TOOL_BATCH_REQUESTS_BUILT total={len(requests)}")
    print(f"WROTE {REQUESTS_PATH}")
    print(f"WROTE {META_PATH}")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.create(requests=requests)
    state = to_plain(batch)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"TOOL_BATCH_SUBMITTED batch_id={state.get('id')} status={state.get('processing_status')}")
    return state.get("id")

def poll(batch_id, sleep_s):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        state = to_plain(batch)
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        counts = state.get("request_counts", {})
        print(
            f"TOOL_BATCH_STATUS id={batch_id} status={state.get('processing_status')} "
            f"processing={counts.get('processing')} succeeded={counts.get('succeeded')} "
            f"errored={counts.get('errored')} canceled={counts.get('canceled')} expired={counts.get('expired')}",
            flush=True,
        )
        if state.get("processing_status") == "ended":
            return state
        time.sleep(sleep_s)

def download(batch_id):
    if RAW_RESULTS_PATH.exists() and RAW_RESULTS_PATH.stat().st_size > 0:
        print(f"RAW_RESULTS_ALREADY_EXISTS {RAW_RESULTS_PATH}")
        return
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    n = 0
    with RAW_RESULTS_PATH.open("w", encoding="utf-8") as f:
        for item in client.messages.batches.results(batch_id):
            obj = to_plain(item)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"TOOL_RESULT_DOWNLOAD_PROGRESS {n}", flush=True)
    print(f"TOOL_RESULT_DOWNLOAD_DONE n={n} path={RAW_RESULTS_PATH}")

def extract_tool_input(obj):
    result = obj.get("result", {})
    if result.get("type") != "succeeded":
        return {}
    content = result.get("message", {}).get("content", []) or []
    for c in content:
        if isinstance(c, dict) and c.get("type") == "tool_use":
            inp = c.get("input", {})
            return inp if isinstance(inp, dict) else {}
    return {}

def extract_text_fallback(obj):
    result = obj.get("result", {})
    if result.get("type") != "succeeded":
        return ""
    parts = []
    for c in result.get("message", {}).get("content", []) or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(str(c.get("text", "")))
    return "\n".join(parts)

def extract_usage(obj):
    result = obj.get("result", {})
    if result.get("type") != "succeeded":
        return {"input_tokens": 0, "output_tokens": 0}
    usage = result.get("message", {}).get("usage", {}) or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }

def parse_answer_fallback(text):
    m = re.search(r"\b([A-H])\b", str(text).upper())
    return m.group(1) if m else ""

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

def summarize():
    meta = {r["custom_id"]: r for r in read_jsonl(META_PATH)}
    raw = read_jsonl(RAW_RESULTS_PATH)

    qa_rows = []
    space_rows = []
    status_counts = Counter()
    total_in = 0
    total_out = 0

    for obj in raw:
        cid = obj.get("custom_id")
        m = meta.get(cid, {})
        if not m:
            continue

        result_type = obj.get("result", {}).get("type", "unknown")
        status_counts[result_type] += 1
        usage = extract_usage(obj)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]

        inp = extract_tool_input(obj)
        text = extract_text_fallback(obj)

        base = {
            **m,
            "result_type": result_type,
            "raw_text": text,
            "tool_input": inp,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

        if m.get("task") == "qa":
            pred = norm_answer(inp.get("answer"))
            if not pred:
                pred = parse_answer_fallback(text)
            sp = norm_space_label(inp.get("space_label"))
            gold = m.get("gold_answer", "")
            correct = int(pred == gold) if gold and pred else (0 if gold else None)
            qa_rows.append({**base, "pred_answer": pred, "pred_space": sp, "correct": correct})

        elif m.get("task") == "space":
            sp = norm_space_label(inp.get("space_label"))
            gold_sp = m.get("gold_space_norm", "")
            correct = int(sp == gold_sp) if gold_sp and sp else (0 if gold_sp else None)
            space_rows.append({**base, "pred_space": sp, "space_correct": correct})

    write_jsonl(OUT / "claude_haiku_4_5_tool_batch_qa.jsonl", qa_rows)
    write_jsonl(OUT / "claude_haiku_4_5_tool_batch_space_v4.jsonl", space_rows)

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
        OUT / "table_claude_haiku_tool_by_method.csv",
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
            dataset, _ = key
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
        OUT / "table_claude_haiku_tool_fusion.csv",
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
        OUT / "table_claude_haiku_tool_space_v4_overall.csv",
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
        OUT / "table_claude_haiku_tool_space_v4_by_label.csv",
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
        OUT / "table_claude_haiku_tool_actual_usage_cost.csv",
        usage_rows,
        ["model", "requests", "input_tokens", "output_tokens", "batch_input_usd_per_mtok", "batch_output_usd_per_mtok", "actual_batch_cost_usd"],
    )

    md = OUT / "CLAUDE_HAIKU_TOOL_BATCH_SUMMARY.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Claude Haiku 4.5 Tool Batch Results\n\n")
        f.write(f"Model: `{MODEL}`\n\n")
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

    print("=== TOOL QA BY METHOD ===")
    for r in by_method:
        print(r)
    print("\n=== TOOL FUSION ===")
    for r in fusion_rows:
        print(r)
    print("\n=== TOOL SPACE OVERALL ===")
    for r in space_overall:
        print(r)
    print("\n=== TOOL SPACE BY LABEL ===")
    for r in space_label_rows:
        print(r)
    print("\n=== TOOL USAGE COST ===")
    for r in usage_rows:
        print(r)

    print(f"\nWROTE {md}")
    print("CLAUDE_HAIKU_TOOL_BATCH_SUMMARY_OK")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--qa-max-per-dataset", type=int, default=1000)
    common.add_argument("--space-max-rows", type=int, default=0)
    common.add_argument("--modes", default="direct,cot,hypmed_v4")
    common.add_argument("--force-new", action="store_true")

    sub.add_parser("submit", parents=[common])
    p = sub.add_parser("poll")
    p.add_argument("--batch-id", default="")
    p.add_argument("--sleep", type=int, default=60)
    p = sub.add_parser("download")
    p.add_argument("--batch-id", default="")
    sub.add_parser("summarize")
    p = sub.add_parser("run-all", parents=[common])
    p.add_argument("--sleep", type=int, default=60)

    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing.")

    if args.cmd == "submit":
        submit(args)
    elif args.cmd == "poll":
        bid = args.batch_id or json.loads(STATE_PATH.read_text(encoding="utf-8"))["id"]
        poll(bid, args.sleep)
    elif args.cmd == "download":
        bid = args.batch_id or json.loads(STATE_PATH.read_text(encoding="utf-8"))["id"]
        download(bid)
    elif args.cmd == "summarize":
        summarize()
    elif args.cmd == "run-all":
        bid = submit(args)
        poll(bid, args.sleep)
        download(bid)
        summarize()
        print("CLAUDE_HAIKU_TOOL_BATCH_RUN_OK")

if __name__ == "__main__":
    main()
