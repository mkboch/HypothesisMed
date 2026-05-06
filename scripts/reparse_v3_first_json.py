import json
import re
import shutil
from pathlib import Path

def normalize(text):
    text = text or ""
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")

def first_json(text):
    text = normalize(text)
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def fallback_answer(text):
    text = normalize(text)
    patterns = [
        r'"answer"\s*:\s*"([A-E])"',
        r"FINAL_ANSWER\s*[:=]\s*([A-E])\b",
        r"Final\s+answer\s*[:=]?\s*([A-E])\b",
        r"\bAnswer\s*[:=]\s*([A-E])\b",
        r"\bthe\s+answer\s+is\s+([A-E])\b",
        r"\bOption\s+([A-E])\b",
    ]
    hits = []
    for pat in patterns:
        hits += re.findall(pat, text, flags=re.I)
    return hits[0].upper() if hits else None

def fallback_space(text):
    text = normalize(text)
    hits = re.findall(r'"space_label"\s*:\s*"(VALID|INCOMPLETE|CONTRADICTED)"', text, flags=re.I)
    if not hits:
        hits = re.findall(r"SPACE_LABEL\s*[:=]\s*(VALID|INCOMPLETE|CONTRADICTED)", text, flags=re.I)
    return hits[0].upper() if hits else None

def fallback_conf(text):
    text = normalize(text)
    hits = re.findall(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text, flags=re.I)
    if not hits:
        hits = re.findall(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.I)
    if hits:
        try:
            return max(0.0, min(1.0, float(hits[0])))
        except Exception:
            return 0.0
    return 0.0

files = sorted(Path("results").glob("qwen2_5_7b_instruct_hypothesismed_v3*medqa_original1000*.jsonl"))

for p in files:
    if "smoke" in p.name:
        continue
    if not p.exists() or p.stat().st_size == 0:
        continue

    backup = p.with_suffix(p.suffix + ".before_first_json_reparse")
    if not backup.exists():
        shutil.copy2(p, backup)

    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    before_missing = sum(r.get("pred_answer") is None for r in rows)

    for r in rows:
        raw = r.get("raw_output", "")
        obj = first_json(raw)

        ans = None
        space = None
        conf = None

        if isinstance(obj, dict):
            ans = obj.get("answer") or obj.get("FINAL_ANSWER") or obj.get("final_answer")
            space = obj.get("space_label") or obj.get("SPACE_LABEL")
            conf = obj.get("confidence") or obj.get("CONFIDENCE")

        if isinstance(ans, str) and ans.strip().upper() in set("ABCDE"):
            r["pred_answer"] = ans.strip().upper()
        else:
            fa = fallback_answer(raw)
            if fa:
                r["pred_answer"] = fa

        if isinstance(space, str) and space.strip().upper() in {"VALID", "INCOMPLETE", "CONTRADICTED"}:
            r["pred_space_label"] = space.strip().upper()
        else:
            fs = fallback_space(raw)
            if fs:
                r["pred_space_label"] = fs

        try:
            r["confidence"] = max(0.0, min(1.0, float(conf)))
        except Exception:
            r["confidence"] = fallback_conf(raw)

    after_missing = sum(r.get("pred_answer") is None for r in rows)

    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = sum(r.get("pred_answer") == r.get("gold_answer") for r in rows) / len(rows)
    print(json.dumps({
        "file": str(p),
        "n": len(rows),
        "missing_before": before_missing,
        "missing_after": after_missing,
        "accuracy_after_first_json_reparse": round(acc, 4)
    }, indent=2))
