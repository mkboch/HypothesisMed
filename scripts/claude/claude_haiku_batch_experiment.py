#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

ROOT = Path("/home/manikm/HypothesisMed")
PHASE2 = ROOT / "results" / "expanded_fix_vllm"
OUT = ROOT / "results" / "expanded_claude_batch"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "claude-haiku-4-5-20251001"
MODEL_LABEL = "claude_haiku_4_5_batch"
LETTERS = "ABCDEFGH"

STATE_PATH = OUT / "claude_haiku_batch_state.json"
META_PATH = OUT / "claude_haiku_batch_metadata.jsonl"
REQUESTS_PATH = OUT / "claude_haiku_batch_requests.jsonl"
RAW_RESULTS_PATH = OUT / "claude_haiku_batch_raw_results.jsonl"

OUTPUT_PLAN = {
    "direct": {"expected_output_tokens": 32, "max_output_tokens": 48},
    "cot": {"expected_output_tokens": 32, "max_output_tokens": 48},
    "hypmed_v4": {"expected_output_tokens": 48, "max_output_tokens": 96},
    "space_v4": {"expected_output_tokens": 32, "max_output_tokens": 64},
}

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
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
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def parse_options(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            kk = str(k).strip().upper()[:1]
            if kk in LETTERS:
                out[kk] = str(v)
        return out
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

def clean_text(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def norm_answer(value, options=None):
    opts = parse_options(options or {})
    if value is None:
        return ""
    if isinstance(value, bool):
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
    if us in LETTERS:
        return us
    m = re.match(r"^\s*([A-H])[\.\)\:\-]\s*", us)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-H])\b", us)
    if m and len(us) <= 20:
        return m.group(1)
    if us.isdigit():
        n = int(us)
        if 0 <= n < len(LETTERS):
            return LETTERS[n]
        if 1 <= n <= len(LETTERS):
            return LETTERS[n - 1]
    cs = clean_text(s)
    for k, v in opts.items():
        if clean_text(v) == cs:
            return k
    for k, v in opts.items():
        if cs and (cs in clean_text(v) or clean_text(v) in cs):
            return k
    return ""

def extract_gold_answer(row):
    keys = [
        "gold_answer", "answer", "gold", "label", "target", "correct",
        "correct_answer", "correct_option", "answer_idx", "answer_index",
        "correct_idx", "correct_index", "label_idx"
    ]
    for k in keys:
        if k in row:
            ans = norm_answer(row.get(k), row.get("options", {}))
            if ans:
                return ans
    return ""

def norm_space_label(x):
    if x is None:
        return ""
    s = str(x).upper()
    if "CONTRADICT" in s:
        return "CONTRADICTED"
    if "INCOMPLETE" in s or "MISSING" in s:
        return "INCOMPLETE"
    if "VALID" in s:
        return "VALID"
    return ""

def extract_gold_space(row):
    keys = ["gold_space_norm", "gold_space", "space_gold", "target_space", "space_label_gold", "label"]
    for k in keys:
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

def prompt_direct(row):
    return f"""You are answering a biomedical multiple-choice question.

Return only one JSON object:
{{"answer":"A|B|C|D|E","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def prompt_cot(row):
    return f"""You are answering a biomedical multiple-choice question.

Reason internally, but do not show reasoning. Return only one JSON object:
{{"answer":"A|B|C|D|E","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def prompt_hypmed_v4(row):
    return f"""You are answering and auditing a biomedical multiple-choice question.

First evaluate the answer space:
VALID: exactly one medically supported best answer is present.
INCOMPLETE: the medically correct answer is missing or insufficiently represented.
CONTRADICTED: options are duplicated, non-unique, mutually inconsistent, or prevent one best answer.

Then select the best answer if possible.

Return only one JSON object:
{{"space_label":"VALID|INCOMPLETE|CONTRADICTED","answer":"A|B|C|D|E","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def prompt_space_v4(row):
    return f"""You are auditing the answer-space validity of a biomedical multiple-choice question.

Use exactly one SPACE label:
VALID: exactly one medically supported best answer is present.
INCOMPLETE: the medically correct answer is missing or insufficiently represented.
CONTRADICTED: options are duplicated, non-unique, mutually inconsistent, or prevent one best answer.

Checklist:
1. Check whether any options are duplicates or near-duplicates.
2. Check whether the option set is non-unique.
3. Check whether the medically correct answer appears missing.
4. Decide whether exactly one best answer is present.

Return only one JSON object:
{{"space_label":"VALID|INCOMPLETE|CONTRADICTED","confidence":0.0}}

Question:
{row.get("question","")}

Options:
{fmt_options(row.get("options",{}))}
"""

def build_batch_payload(qa_max_per_dataset=1000, space_max_rows=0, modes=None):
    modes = modes or ["direct", "cot", "hypmed_v4"]
    qa_input = PHASE2 / "stronger_model_eval_input.jsonl"
    space_input = PHASE2 / "space_v4_stress_input.jsonl"

    qa_rows = read_jsonl(qa_input)
    space_rows = read_jsonl(space_input)

    by_dataset = defaultdict(int)
    kept_qa = []
    for i, row in enumerate(qa_rows):
        d = str(row.get("dataset", "unknown"))
        if qa_max_per_dataset > 0 and by_dataset[d] >= qa_max_per_dataset:
            continue
        by_dataset[d] += 1
        kept_qa.append((i, row))

    prompt_map = {
        "direct": prompt_direct,
        "cot": prompt_cot,
        "hypmed_v4": prompt_hypmed_v4,
    }

    requests = []
    metadata = []
    seq = 0

    for original_index, row in kept_qa:
        dataset = str(row.get("dataset", "unknown"))
        rid = row_id(row, original_index)
        gold_answer = extract_gold_answer(row)
        for mode in modes:
            seq += 1
            custom_id = f"r{seq:07d}"
            prompt = prompt_map[mode](row)
            max_tokens = OUTPUT_PLAN[mode]["max_output_tokens"]
            req = {
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            meta = {
                "custom_id": custom_id,
                "task": "qa",
                "model": MODEL_LABEL,
                "dataset": dataset,
                "method": mode,
                "item_id": f"{dataset}:{rid}",
                "row_id": rid,
                "original_index": original_index,
                "gold_answer": gold_answer,
                "gold_space_norm": "",
                "max_tokens": max_tokens,
                "prompt": prompt,
            }
            requests.append(req)
            metadata.append(meta)

    if space_max_rows > 0:
        space_rows = space_rows[:space_max_rows]

    for i, row in enumerate(space_rows):
        seq += 1
        custom_id = f"r{seq:07d}"
        dataset = str(row.get("dataset", "stress"))
        rid = row_id(row, i)
        prompt = prompt_space_v4(row)
        max_tokens = OUTPUT_PLAN["space_v4"]["max_output_tokens"]
        req = {
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        meta = {
            "custom_id": custom_id,
            "task": "space",
            "model": MODEL_LABEL,
            "dataset": dataset,
            "method": "space_v4",
            "item_id": f"{dataset}:{rid}",
            "row_id": rid,
            "original_index": i,
            "gold_answer": "",
            "gold_space_norm": extract_gold_space(row),
            "max_tokens": max_tokens,
            "prompt": prompt,
        }
        requests.append(req)
        metadata.append(meta)

    return requests, metadata

def to_plain(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if hasattr(x, "dict"):
        return x.dict()
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"raw": str(x)}

def check_budget_or_die(cap=3.25):
    p = ROOT / "results" / "claude_cost_estimate" / "table_claude_cost_estimate.csv"
    if not p.exists():
        raise SystemExit(f"Missing cost estimate table: {p}")
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    target = None
    for r in rows:
        if r.get("pricing") == "haiku_4_5_batch":
            target = r
            break
    if not target:
        raise SystemExit("Missing haiku_4_5_batch row in cost estimate.")
    cap_cost = float(target["estimated_cost_output_cap_usd"])
    expected = float(target["estimated_cost_expected_output_usd"])
    print(f"BUDGET_CHECK haiku_4_5_batch expected=${expected:.4f} cap=${cap_cost:.4f} allowed_cap=${cap:.4f}")
    if cap_cost > cap:
        raise SystemExit(f"BUDGET_ABORT cap_cost ${cap_cost:.4f} > allowed ${cap:.4f}")

def submit_batch(args):
    if STATE_PATH.exists() and not args.force_new:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        print(f"EXISTING_BATCH_STATE batch_id={state.get('id')} status={state.get('processing_status')}")
        return state.get("id")

    check_budget_or_die(args.budget_cap)

    requests, metadata = build_batch_payload(
        qa_max_per_dataset=args.qa_max_per_dataset,
        space_max_rows=args.space_max_rows,
        modes=[m.strip() for m in args.modes.split(",") if m.strip()],
    )

    write_jsonl(META_PATH, metadata)
    write_jsonl(REQUESTS_PATH, requests)

    print(f"BATCH_REQUESTS_BUILT total={len(requests)}")
    print(f"WROTE {META_PATH}")
    print(f"WROTE {REQUESTS_PATH}")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    batch = client.messages.batches.create(requests=requests)
    state = to_plain(batch)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"BATCH_SUBMITTED batch_id={state.get('id')} status={state.get('processing_status')}")
    print(f"WROTE {STATE_PATH}")
    return state.get("id")

def poll_batch(batch_id, sleep_s=60):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        state = to_plain(batch)
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        counts = state.get("request_counts", {})
        print(
            "BATCH_STATUS "
            f"id={batch_id} status={state.get('processing_status')} "
            f"processing={counts.get('processing')} succeeded={counts.get('succeeded')} "
            f"errored={counts.get('errored')} canceled={counts.get('canceled')} expired={counts.get('expired')}",
            flush=True
        )
        if state.get("processing_status") == "ended":
            return state
        time.sleep(sleep_s)

def download_results(batch_id):
    if RAW_RESULTS_PATH.exists() and RAW_RESULTS_PATH.stat().st_size > 0:
        print(f"RAW_RESULTS_ALREADY_EXISTS {RAW_RESULTS_PATH}")
        return

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    n = 0
    with RAW_RESULTS_PATH.open("w", encoding="utf-8") as f:
        for item in client.messages.batches.results(batch_id):
            obj = to_plain(item)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"RESULT_DOWNLOAD_PROGRESS {n}", flush=True)

    print(f"RESULT_DOWNLOAD_DONE n={n} path={RAW_RESULTS_PATH}")

def parse_json_object(text):
    if not text:
        return {}
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}

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

def parse_answer(text):
    obj = parse_json_object(text)
    for k in ["answer", "predicted_answer", "final_answer", "choice", "selected_answer"]:
        if k in obj:
            ans = norm_answer(obj.get(k), {})
            if ans:
                return ans
    m = re.search(r'"(?:answer|predicted_answer|final_answer|choice)"\s*:\s*"?(A|B|C|D|E|F|G|H)"?', text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(?:answer|choice|final)\b[^A-H]{0,20}\b([A-H])\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    return ""

def parse_space(text):
    obj = parse_json_object(text)
    for k in ["space_label", "space", "label", "answer_space", "space_status"]:
        if k in obj:
            y = norm_space_label(obj.get(k))
            if y:
                return y
    m = re.search(r'"(?:space_label|space|label)"\s*:\s*"?(VALID|INCOMPLETE|CONTRADICTED)"?', text, flags=re.I)
    if m:
        return norm_space_label(m.group(1))
    return norm_space_label(text)

def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def safe_acc(corrects):
    vals = [x for x in corrects if x is not None]
    if not vals:
        return ""
    return round(sum(vals) / len(vals), 6)

def summarize():
    if not META_PATH.exists():
        raise SystemExit(f"Missing {META_PATH}")
    if not RAW_RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RAW_RESULTS_PATH}")

    meta = {}
    for r in read_jsonl(META_PATH):
        meta[r["custom_id"]] = r

    qa_rows = []
    space_rows = []
    status_counts = Counter()
    total_in = 0
    total_out = 0

    for obj in read_jsonl(RAW_RESULTS_PATH):
        cid = obj.get("custom_id")
        m = meta.get(cid, {})
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
            pred = parse_answer(text)
            sp = parse_space(text)
            gold = m.get("gold_answer", "")
            correct = None
            if gold:
                correct = int(pred == gold)
            qa_rows.append({
                **base,
                "pred_answer": pred,
                "pred_space": sp,
                "correct": correct,
            })
        elif m.get("task") == "space":
            sp = parse_space(text)
            gold_sp = m.get("gold_space_norm", "")
            correct = None
            if gold_sp:
                correct = int(sp == gold_sp)
            space_rows.append({
                **base,
                "pred_space": sp,
                "space_correct": correct,
            })

    qa_out = OUT / "claude_haiku_4_5_batch_qa.jsonl"
    space_out = OUT / "claude_haiku_4_5_batch_space_v4.jsonl"
    write_jsonl(qa_out, qa_rows)
    write_jsonl(space_out, space_rows)

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
        OUT / "table_claude_haiku_by_method.csv",
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

    def fusion_pick(preds, order):
        vals = [preds.get(k, "") for k in order if preds.get(k, "")]
        if not vals:
            return ""
        c = Counter(vals)
        top_count = max(c.values())
        winners = {k for k, v in c.items() if v == top_count}
        for k in order:
            if preds.get(k, "") in winners:
                return preds.get(k, "")
        return vals[0]

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
            correct = None
            if gold:
                correct = int(pred == gold)
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
        OUT / "table_claude_haiku_fusion.csv",
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
        OUT / "table_claude_haiku_space_v4_overall.csv",
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
        OUT / "table_claude_haiku_space_v4_by_label.csv",
        space_label_rows,
        ["model", "gold_space_norm", "N", "space_coverage", "space_v4_accuracy", "hybrid_space_accuracy"],
    )

    expected_cost = total_in / 1_000_000 * 0.50 + total_out / 1_000_000 * 2.50
    usage_rows = [{
        "model": MODEL_LABEL,
        "requests": len(qa_rows) + len(space_rows),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "batch_input_usd_per_mtok": 0.50,
        "batch_output_usd_per_mtok": 2.50,
        "actual_batch_cost_usd": round(expected_cost, 6),
    }]
    write_csv(
        OUT / "table_claude_haiku_actual_usage_cost.csv",
        usage_rows,
        ["model", "requests", "input_tokens", "output_tokens", "batch_input_usd_per_mtok", "batch_output_usd_per_mtok", "actual_batch_cost_usd"],
    )

    md = OUT / "CLAUDE_HAIKU_BATCH_SUMMARY.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Claude Haiku 4.5 Batch Results\n\n")
        f.write(f"Model: `{MODEL}`\n\n")
        f.write(f"Result status counts: `{dict(status_counts)}`\n\n")
        f.write(f"QA rows: **{len(qa_rows)}**\n\n")
        f.write(f"SPACE rows: **{len(space_rows)}**\n\n")
        f.write(f"Actual usage: input tokens **{total_in:,}**, output tokens **{total_out:,}**\n\n")
        f.write(f"Estimated actual batch cost: **${expected_cost:.4f}**\n\n")
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

    print(f"WROTE {qa_out}")
    print(f"WROTE {space_out}")
    print(f"WROTE {OUT / 'table_claude_haiku_by_method.csv'}")
    print(f"WROTE {OUT / 'table_claude_haiku_fusion.csv'}")
    print(f"WROTE {OUT / 'table_claude_haiku_space_v4_overall.csv'}")
    print(f"WROTE {OUT / 'table_claude_haiku_space_v4_by_label.csv'}")
    print(f"WROTE {OUT / 'table_claude_haiku_actual_usage_cost.csv'}")
    print(f"WROTE {md}")
    print("CLAUDE_HAIKU_BATCH_SUMMARY_OK")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--qa-max-per-dataset", type=int, default=1000)
    common.add_argument("--space-max-rows", type=int, default=0)
    common.add_argument("--modes", default="direct,cot,hypmed_v4")
    common.add_argument("--budget-cap", type=float, default=3.25)
    common.add_argument("--force-new", action="store_true")

    p = sub.add_parser("submit", parents=[common])
    p = sub.add_parser("poll")
    p.add_argument("--batch-id", default="")
    p.add_argument("--sleep", type=int, default=60)
    p = sub.add_parser("download")
    p.add_argument("--batch-id", default="")
    p = sub.add_parser("summarize")
    p = sub.add_parser("run-all", parents=[common])
    p.add_argument("--sleep", type=int, default=60)

    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing.")

    if args.cmd == "submit":
        submit_batch(args)
    elif args.cmd == "poll":
        bid = args.batch_id or json.loads(STATE_PATH.read_text(encoding="utf-8"))["id"]
        poll_batch(bid, sleep_s=args.sleep)
    elif args.cmd == "download":
        bid = args.batch_id or json.loads(STATE_PATH.read_text(encoding="utf-8"))["id"]
        download_results(bid)
    elif args.cmd == "summarize":
        summarize()
    elif args.cmd == "run-all":
        bid = submit_batch(args)
        poll_batch(bid, sleep_s=args.sleep)
        download_results(bid)
        summarize()
        print("CLAUDE_HAIKU_BATCH_RUN_OK")

if __name__ == "__main__":
    main()
