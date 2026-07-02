#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/manikm/HypothesisMed")
OUT = ROOT / "results" / "expanded_claude_batch"
META = OUT / "claude_haiku_batch_metadata.jsonl"
RAW = OUT / "claude_haiku_batch_raw_results.jsonl"
ROBUST = OUT / "claude_haiku_4_5_batch_qa_robust.jsonl"

def read_jsonl(p):
    rows = []
    with Path(p).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def get_text(obj):
    result = obj.get("result", {})
    if result.get("type") != "succeeded":
        return ""
    msg = result.get("message", {})
    content = msg.get("content", [])
    return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text").strip()

def get_stop_reason(obj):
    result = obj.get("result", {})
    if result.get("type") != "succeeded":
        return result.get("type", "not_succeeded")
    return result.get("message", {}).get("stop_reason", "")

def get_usage(obj):
    result = obj.get("result", {})
    msg = result.get("message", {}) if isinstance(result, dict) else {}
    usage = msg.get("usage", {}) or {}
    return int(usage.get("output_tokens") or 0)

meta = {r["custom_id"]: r for r in read_jsonl(META)}
raw = {r["custom_id"]: r for r in read_jsonl(RAW)}
qa = read_jsonl(ROBUST)

group = defaultdict(list)
for r in qa:
    group[(r["dataset"], r["method"])].append(r)

lines = []
lines.append("# Claude Haiku parse diagnostic\n")

for key, rows in sorted(group.items()):
    dataset, method = key
    N = len(rows)
    parsed = [r for r in rows if r.get("pred_answer")]
    unparsed = [r for r in rows if not r.get("pred_answer")]
    correct = sum(1 for r in rows if r.get("correct") == 1)
    stop_counts = Counter()
    out_tok = []
    unparsed_stop_counts = Counter()

    for r in rows:
        obj = raw.get(r["custom_id"], {})
        stop = get_stop_reason(obj)
        stop_counts[stop] += 1
        out_tok.append(get_usage(obj))
        if not r.get("pred_answer"):
            unparsed_stop_counts[stop] += 1

    avg_out = sum(out_tok) / max(1, len(out_tok))
    lines.append(f"\n## {dataset} / {method}")
    lines.append(f"N={N} parsed={len(parsed)} parse_coverage={len(parsed)/N:.3f} accuracy_all={correct/N:.3f}")
    lines.append(f"stop_counts={dict(stop_counts)}")
    lines.append(f"unparsed_stop_counts={dict(unparsed_stop_counts)}")
    lines.append(f"avg_output_tokens={avg_out:.1f}")

    lines.append("\n### First 5 unparsed examples")
    for r in unparsed[:5]:
        obj = raw.get(r["custom_id"], {})
        m = meta.get(r["custom_id"], {})
        text = get_text(obj)
        prompt = m.get("prompt", "")
        q = ""
        if "Question:" in prompt and "Options:" in prompt:
            q = prompt.split("Question:", 1)[1].split("Options:", 1)[0].strip()
        opts = prompt.split("Options:", 1)[1].strip() if "Options:" in prompt else ""
        lines.append("\n```json")
        lines.append(json.dumps({
            "custom_id": r["custom_id"],
            "dataset": dataset,
            "method": method,
            "gold_answer": r.get("gold_answer"),
            "stop_reason": get_stop_reason(obj),
            "output_tokens": get_usage(obj),
            "question_start": q[:500],
            "options_start": opts[:800],
            "raw_text": text[:1500],
        }, indent=2, ensure_ascii=False))
        lines.append("```")

text = "\n".join(lines)
p = OUT / "CLAUDE_PARSE_DIAGNOSTIC.md"
p.write_text(text + "\n", encoding="utf-8")
print(text[:12000])
print(f"\nWROTE {p}")
print("CLAUDE_PARSE_DIAGNOSTIC_OK")
