import json
import re

def _clean_json_block(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    return text

def extract_json(text):
    if not text:
        return {}

    candidates = []

    # fenced json blocks
    candidates += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.DOTALL)

    # any json-like object
    candidates += re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)

    # try from last to first because final answer usually appears near the end
    for cand in reversed(candidates):
        cand = _clean_json_block(cand)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {}

def extract_answer(obj, raw_text=None):
    if isinstance(obj, dict):
        for key in ["final_answer", "answer", "pred_answer"]:
            if key in obj and obj[key] is not None:
                m = re.search(r"\b([A-E])\b", str(obj[key]).upper())
                if m:
                    return m.group(1)

    text = (raw_text or "").upper()

    patterns = [
        r"FINAL_ANSWER\s*[:=]\s*[\"']?([A-E])\b",
        r"ANSWER\s*[:=]\s*[\"']?([A-E])\b",
        r'"ANSWER"\s*:\s*"([A-E])"',
        r'"FINAL_ANSWER"\s*:\s*"([A-E])"',
        r"FINAL ANSWER\s*[:=]\s*([A-E])\b",
        r"THE ANSWER IS\s*([A-E])\b",
        r"ANSWER IS\s*([A-E])\b",
    ]

    for pat in patterns:
        matches = re.findall(pat, text, flags=re.I)
        if matches:
            return matches[-1]

    return None

def extract_space_label(obj, raw_text=None):
    if isinstance(obj, dict):
        for key in ["space_label", "SPACE_LABEL", "label"]:
            val = obj.get(key)
            if val:
                val = str(val).strip().upper()
                if val in {"VALID", "INCOMPLETE", "CONTRADICTED"}:
                    return val

    text = (raw_text or "").upper()

    patterns = [
        r"SPACE_LABEL\s*[:=]\s*(VALID|INCOMPLETE|CONTRADICTED)",
        r"SPACE LABEL\s*[:=]\s*(VALID|INCOMPLETE|CONTRADICTED)",
        r'"SPACE_LABEL"\s*:\s*"(VALID|INCOMPLETE|CONTRADICTED)"',
    ]

    for pat in patterns:
        matches = re.findall(pat, text, flags=re.I)
        if matches:
            return matches[-1].upper()

    return None

def extract_confidence(obj, raw_text=None):
    if isinstance(obj, dict):
        for key in ["confidence", "CONFIDENCE"]:
            if key in obj:
                try:
                    return float(obj[key])
                except Exception:
                    pass

    text = raw_text or ""
    matches = re.findall(r"CONFIDENCE\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.I)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
            pass

    matches = re.findall(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text, flags=re.I)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
            pass

    return 0.0
