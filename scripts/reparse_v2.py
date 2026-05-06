import argparse, json, re
from pathlib import Path

def last_json_object(text):
    if not text:
        return None
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except Exception:
            pass
    return None

def extract_answer(text, options):
    allowed = {str(k).strip().upper() for k in options.keys()}
    obj = last_json_object(text)
    if isinstance(obj, dict):
        for key in ["answer", "FINAL_ANSWER", "final_answer"]:
            val = obj.get(key)
            if isinstance(val, str):
                val = val.strip().upper()
                if val in allowed:
                    return val

    patterns = [
        r'"answer"\s*:\s*"([A-E])"',
        r"FINAL_ANSWER\s*[:=]\s*([A-E])\b",
        r"Final answer\s*[:=]?\s*([A-E])\b",
        r"\banswer is\s+([A-E])\b",
        r"\bOption\s+([A-E])\b",
    ]
    hits = []
    for pat in patterns:
        hits += re.findall(pat, text or "", flags=re.I)
    hits = [h.upper() for h in hits if h.upper() in allowed]
    if hits:
        return hits[-1]

    if "none of the above" in (text or "").lower() and "E" in allowed:
        return "E"
    return None

def extract_space(text):
    obj = last_json_object(text)
    if isinstance(obj, dict):
        for key in ["space_label", "SPACE_LABEL"]:
            val = obj.get(key)
            if isinstance(val, str) and val.strip().upper() in {"VALID","INCOMPLETE","CONTRADICTED"}:
                return val.strip().upper()
    hits = re.findall(r"SPACE_LABEL\s*[:=]\s*(VALID|INCOMPLETE|CONTRADICTED)", text or "", flags=re.I)
    if hits:
        return hits[-1].upper()
    hits = re.findall(r'"space_label"\s*:\s*"(VALID|INCOMPLETE|CONTRADICTED)"', text or "", flags=re.I)
    if hits:
        return hits[-1].upper()
    return None

def extract_conf(text):
    obj = last_json_object(text)
    if isinstance(obj, dict):
        for key in ["confidence", "CONFIDENCE"]:
            try:
                return max(0.0, min(1.0, float(obj.get(key))))
            except Exception:
                pass
    hits = re.findall(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", text or "", flags=re.I)
    if not hits:
        hits = re.findall(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text or "", flags=re.I)
    if hits:
        try:
            return max(0.0, min(1.0, float(hits[-1])))
        except Exception:
            pass
    return 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--infile", required=True)
ap.add_argument("--outdir", default="results/reparsed_v2")
args = ap.parse_args()

inp = Path(args.infile)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / inp.name

n = miss0 = miss1 = 0
with inp.open() as f, out.open("w") as g:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if r.get("pred_answer") is None:
            miss0 += 1
        raw = r.get("raw_output", "")
        r["pred_answer"] = extract_answer(raw, r.get("options", {})) or r.get("pred_answer")
        r["pred_space_label"] = extract_space(raw) or r.get("pred_space_label")
        r["confidence"] = extract_conf(raw)
        if r.get("pred_answer") is None:
            miss1 += 1
        g.write(json.dumps(r, ensure_ascii=False) + "\n")

print(json.dumps({"input": str(inp), "output": str(out), "n": n, "missing_before": miss0, "missing_after": miss1}, indent=2))
