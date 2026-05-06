import json
import re

VALID_SPACES = {"VALID", "INCOMPLETE", "CONTRADICTED"}

def normalize_text(text):
    text = text or ""
    text = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return text

def _first_json_object(text):
    text = normalize_text(text)
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def parse_output(text):
    text = normalize_text(text)
    out = {}

    obj = _first_json_object(text)
    if isinstance(obj, dict):
        ans = obj.get("answer") or obj.get("FINAL_ANSWER") or obj.get("final_answer")
        sp = obj.get("space_label") or obj.get("SPACE_LABEL")
        conf = obj.get("confidence") or obj.get("CONFIDENCE")

        if isinstance(ans, str):
            ans = ans.strip().upper()
            if re.fullmatch(r"[A-E]", ans):
                out["answer"] = ans

        if isinstance(sp, str):
            sp = sp.strip().upper()
            if sp in VALID_SPACES:
                out["space_label"] = sp

        try:
            out["confidence"] = max(0.0, min(1.0, float(conf)))
        except Exception:
            pass

    if "answer" not in out:
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
        if hits:
            out["answer"] = hits[0].upper()

    if "space_label" not in out:
        hits = re.findall(r'"space_label"\s*:\s*"(VALID|INCOMPLETE|CONTRADICTED)"', text, flags=re.I)
        if not hits:
            hits = re.findall(r"SPACE_LABEL\s*[:=]\s*(VALID|INCOMPLETE|CONTRADICTED)", text, flags=re.I)
        if hits:
            out["space_label"] = hits[0].upper()

    if "confidence" not in out:
        hits = re.findall(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text, flags=re.I)
        if not hits:
            hits = re.findall(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.I)
        if hits:
            try:
                out["confidence"] = max(0.0, min(1.0, float(hits[0])))
            except Exception:
                out["confidence"] = 0.0

    if "confidence" not in out:
        out["confidence"] = 0.0

    return out
