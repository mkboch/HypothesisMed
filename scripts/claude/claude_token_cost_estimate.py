#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic

ROOT = Path("/home/manikm/HypothesisMed")
PHASE2 = ROOT / "results" / "expanded_fix_vllm"
OUT = ROOT / "results" / "claude_cost_estimate"
OUT.mkdir(parents=True, exist_ok=True)

LETTERS = "ABCDEFGH"

# Prices are USD per million tokens.
PRICES = [
    {"pricing": "haiku_4_5_standard", "input_per_mtok": 1.00, "output_per_mtok": 5.00},
    {"pricing": "haiku_4_5_batch", "input_per_mtok": 0.50, "output_per_mtok": 2.50},
    {"pricing": "sonnet_4_6_standard", "input_per_mtok": 3.00, "output_per_mtok": 15.00},
    {"pricing": "sonnet_4_6_batch", "input_per_mtok": 1.50, "output_per_mtok": 7.50},
]

# Expected output is our realistic JSON-only estimate.
# Max output is the hard cap we should use in the actual Claude run later.
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

def parse_options(x):
    if isinstance(x, dict):
        return {str(k).upper()[:1]: str(v) for k, v in x.items()}
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

def safe_id(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))[:80]

def build_requests(qa_max_per_dataset, space_max_rows, modes):
    qa_input = PHASE2 / "stronger_model_eval_input.jsonl"
    space_input = PHASE2 / "space_v4_stress_input.jsonl"

    if not qa_input.exists():
        raise SystemExit(f"Missing {qa_input}")
    if not space_input.exists():
        raise SystemExit(f"Missing {space_input}")

    requests = []

    qa_rows = read_jsonl(qa_input)
    by_dataset = defaultdict(int)
    kept = []
    for row in qa_rows:
        d = str(row.get("dataset", "unknown"))
        if qa_max_per_dataset > 0 and by_dataset[d] >= qa_max_per_dataset:
            continue
        by_dataset[d] += 1
        kept.append(row)

    prompt_map = {
        "direct": prompt_direct,
        "cot": prompt_cot,
        "hypmed_v4": prompt_hypmed_v4,
    }

    for row in kept:
        dataset = str(row.get("dataset", "unknown"))
        rid = str(row.get("id", ""))
        for mode in modes:
            if mode not in prompt_map:
                raise SystemExit(f"Unknown mode: {mode}")
            prompt = prompt_map[mode](row)
            plan = OUTPUT_PLAN[mode]
            requests.append({
                "custom_id": f"qa_{safe_id(dataset)}_{safe_id(rid)}_{mode}",
                "task": "qa",
                "dataset": dataset,
                "method": mode,
                "prompt": prompt,
                "expected_output_tokens": plan["expected_output_tokens"],
                "max_output_tokens": plan["max_output_tokens"],
            })

    space_rows = read_jsonl(space_input)
    if space_max_rows > 0:
        space_rows = space_rows[:space_max_rows]

    for i, row in enumerate(space_rows):
        dataset = str(row.get("dataset", "stress"))
        rid = str(row.get("id", i))
        prompt = prompt_space_v4(row)
        plan = OUTPUT_PLAN["space_v4"]
        requests.append({
            "custom_id": f"space_{safe_id(dataset)}_{safe_id(rid)}_{i}",
            "task": "space",
            "dataset": dataset,
            "method": "space_v4",
            "prompt": prompt,
            "expected_output_tokens": plan["expected_output_tokens"],
            "max_output_tokens": plan["max_output_tokens"],
        })

    return requests

def load_cache(path):
    cache = {}
    p = Path(path)
    if not p.exists():
        return cache
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cache[obj["key"]] = int(obj["input_tokens"])
            except Exception:
                pass
    return cache

def append_cache(path, obj):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def count_prompt_tokens(client, model, prompt):
    last = None
    for attempt in range(12):
        try:
            resp = client.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return int(resp.input_tokens)
        except Exception as e:
            name = e.__class__.__name__.lower()
            msg = repr(e)
            last = msg
            if "authentication" in name or "permission" in name or "notfound" in name or "badrequest" in name:
                raise
            sleep_s = min(90, 2 ** attempt) + random.random()
            print(f"COUNT_RETRY attempt={attempt+1} sleep={sleep_s:.1f}s error={msg[:240]}", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"count_tokens failed after retries: {last}")

def md_table(headers, rows):
    s = []
    s.append("| " + " | ".join(headers) + " |")
    s.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        s.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--qa-max-per-dataset", type=int, default=1000)
    ap.add_argument("--space-max-rows", type=int, default=0)
    ap.add_argument("--modes", default="direct,cot,hypmed_v4")
    ap.add_argument("--sleep", type=float, default=0.02)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is missing.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    requests = build_requests(args.qa_max_per_dataset, args.space_max_rows, modes)

    print(f"REQUESTS_BUILT total={len(requests)} model_for_count={args.model}")
    print(f"QA_MAX_PER_DATASET={args.qa_max_per_dataset} SPACE_MAX_ROWS={args.space_max_rows or 'ALL'} MODES={','.join(modes)}")

    cache_path = OUT / f"token_count_cache_{safe_id(args.model)}.jsonl"
    cache = load_cache(cache_path)
    client = anthropic.Anthropic(api_key=api_key)

    rows = []
    new_done = 0

    for i, r in enumerate(requests, start=1):
        key = hashlib.sha256((args.model + "\0" + r["prompt"]).encode("utf-8")).hexdigest()
        if key in cache:
            input_tokens = cache[key]
        else:
            input_tokens = count_prompt_tokens(client, args.model, r["prompt"])
            cache[key] = input_tokens
            append_cache(cache_path, {"key": key, "input_tokens": input_tokens})
            new_done += 1
            time.sleep(args.sleep)

        rows.append({
            "custom_id": r["custom_id"],
            "task": r["task"],
            "dataset": r["dataset"],
            "method": r["method"],
            "input_tokens": input_tokens,
            "expected_output_tokens": r["expected_output_tokens"],
            "max_output_tokens": r["max_output_tokens"],
        })

        if i % 100 == 0:
            print(f"TOKEN_COUNT_PROGRESS {i}/{len(requests)} new_api_counts={new_done}", flush=True)

    request_csv = OUT / "claude_token_count_requests.csv"
    with request_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "custom_id", "task", "dataset", "method",
            "input_tokens", "expected_output_tokens", "max_output_tokens"
        ])
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(lambda: {
        "N": 0,
        "input_tokens": 0,
        "expected_output_tokens": 0,
        "max_output_tokens": 0,
    })

    for r in rows:
        key = (r["task"], r["dataset"], r["method"])
        g = groups[key]
        g["N"] += 1
        g["input_tokens"] += int(r["input_tokens"])
        g["expected_output_tokens"] += int(r["expected_output_tokens"])
        g["max_output_tokens"] += int(r["max_output_tokens"])

    summary_csv = OUT / "table_claude_token_summary_by_task.csv"
    summary_rows = []
    for (task, dataset, method), g in sorted(groups.items()):
        avg_in = g["input_tokens"] / max(1, g["N"])
        summary_rows.append({
            "task": task,
            "dataset": dataset,
            "method": method,
            "N": g["N"],
            "input_tokens": g["input_tokens"],
            "avg_input_tokens": round(avg_in, 2),
            "expected_output_tokens": g["expected_output_tokens"],
            "max_output_tokens": g["max_output_tokens"],
        })

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "dataset", "method", "N", "input_tokens", "avg_input_tokens",
            "expected_output_tokens", "max_output_tokens"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    total_input = sum(int(r["input_tokens"]) for r in rows)
    total_expected_output = sum(int(r["expected_output_tokens"]) for r in rows)
    total_max_output = sum(int(r["max_output_tokens"]) for r in rows)
    total_requests = len(rows)

    cost_rows = []
    for p in PRICES:
        expected_cost = (
            total_input / 1_000_000 * p["input_per_mtok"]
            + total_expected_output / 1_000_000 * p["output_per_mtok"]
        )
        max_cost = (
            total_input / 1_000_000 * p["input_per_mtok"]
            + total_max_output / 1_000_000 * p["output_per_mtok"]
        )
        cost_rows.append({
            "pricing": p["pricing"],
            "requests": total_requests,
            "input_tokens_exact": total_input,
            "expected_output_tokens_est": total_expected_output,
            "max_output_tokens_cap": total_max_output,
            "estimated_cost_expected_output_usd": round(expected_cost, 4),
            "estimated_cost_output_cap_usd": round(max_cost, 4),
        })

    cost_csv = OUT / "table_claude_cost_estimate.csv"
    with cost_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pricing", "requests", "input_tokens_exact",
            "expected_output_tokens_est", "max_output_tokens_cap",
            "estimated_cost_expected_output_usd", "estimated_cost_output_cap_usd"
        ])
        writer.writeheader()
        writer.writerows(cost_rows)

    md = OUT / "CLAUDE_COST_ESTIMATE_SUMMARY.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Claude token and cost estimate\n\n")
        f.write(f"Model used for input token counting: `{args.model}`\n\n")
        f.write(f"Total planned requests: **{total_requests}**\n\n")
        f.write(f"Exact counted input tokens: **{total_input:,}**\n\n")
        f.write(f"Estimated JSON-only output tokens: **{total_expected_output:,}**\n\n")
        f.write(f"Hard output-token cap if we run with planned max_tokens: **{total_max_output:,}**\n\n")
        f.write("## Cost estimate\n\n")
        f.write(md_table(
            ["pricing", "requests", "input tokens", "expected output", "max output cap", "expected cost USD", "cap cost USD"],
            [[
                r["pricing"],
                r["requests"],
                f'{r["input_tokens_exact"]:,}',
                f'{r["expected_output_tokens_est"]:,}',
                f'{r["max_output_tokens_cap"]:,}',
                f'${r["estimated_cost_expected_output_usd"]:.4f}',
                f'${r["estimated_cost_output_cap_usd"]:.4f}',
            ] for r in cost_rows]
        ))
        f.write("\n\n")
        f.write("## Interpretation\n\n")
        f.write("- The safest low-cost full-coverage option is usually `haiku_4_5_batch`.\n")
        f.write("- The expected-cost column assumes concise JSON outputs.\n")
        f.write("- The cap-cost column assumes every request reaches the planned `max_tokens`, so it is the safer upper bound for budgeting.\n")
        f.write("- This estimate counts input tokens exactly with Anthropic's token-counting endpoint, but output tokens are estimated until actual generation is run.\n")

    print("\n=== COST ESTIMATE ===")
    for r in cost_rows:
        print(
            f"{r['pricing']}: expected=${r['estimated_cost_expected_output_usd']:.4f} "
            f"cap=${r['estimated_cost_output_cap_usd']:.4f}"
        )

    print(f"\nWROTE {request_csv}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {cost_csv}")
    print(f"WROTE {md}")
    print("CLAUDE_TOKEN_ESTIMATE_OK")

if __name__ == "__main__":
    main()
