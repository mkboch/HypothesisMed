import argparse
import json
import re
from pathlib import Path

LETTERS = set("ABCDE")

def extract_answer(text, options):
    if not text:
        return None
    allowed = set(str(k).strip().upper() for k in options.keys())

    patterns = [
        r"FINAL_ANSWER\s*[:=]\s*([A-E])\b",
        r"Final answer\s*[:=]?\s*([A-E])\b",
        r"answer\s*[:=]\s*['\"]?([A-E])['\"]?",
        r'"answer"\s*:\s*"([A-E])"',
        r'"FINAL_ANSWER"\s*:\s*"([A-E])"',
        r"\bOption\s+([A-E])\b",
        r"\bAnswer\s+is\s+([A-E])\b",
        r"\bThe answer is\s+([A-E])\b",
    ]

    hits = []
    for pat in patterns:
        hits += re.findall(pat, text, flags=re.IGNORECASE)

    hits = [h.upper() for h in hits if h and h.upper() in allowed]
    if hits:
        return hits[-1]

    # fallback: if output says none of the above and E exists
    if "none of the above" in text.lower() and "E" in allowed:
        return "E"

    return None

def extract_space(text):
    if not text:
        return None
    labels = re.findall(r"SPACE_LABEL\s*[:=]\s*(VALID|INCOMPLETE|CONTRADICTED)\b", text, flags=re.IGNORECASE)
    if labels:
        return labels[-1].upper()

    labels = re.findall(r'"space_label"\s*:\s*"(VALID|INCOMPLETE|CONTRADICTED)"', text, flags=re.IGNORECASE)
    if labels:
        return labels[-1].upper()

    return None

def extract_conf(text):
    if not text:
        return 0.0
    vals = re.findall(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not vals:
        vals = re.findall(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text, flags=re.IGNORECASE)
    if vals:
        try:
            return max(0.0, min(1.0, float(vals[-1])))
        except Exception:
            return 0.0
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", default="results/reparsed_paper_ready")
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / infile.name

    n = 0
    missing_before = 0
    missing_after = 0

    with infile.open() as f, outfile.open("w") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1

            if r.get("pred_answer") is None:
                missing_before += 1

            raw = r.get("raw_output", "")
            options = r.get("options", {})

            ans = extract_answer(raw, options)
            space = extract_space(raw)
            conf = extract_conf(raw)

            # Preserve existing if robust parser finds nothing
            if ans is None:
                ans = r.get("pred_answer")
            if space is None:
                space = r.get("pred_space_label")
            if conf == 0.0 and r.get("confidence") is not None:
                try:
                    conf = float(r.get("confidence"))
                except Exception:
                    conf = 0.0

            r["pred_answer"] = ans
            r["pred_space_label"] = space
            r["confidence"] = conf

            if r.get("pred_answer") is None:
                missing_after += 1

            g.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "infile": str(infile),
        "outfile": str(outfile),
        "n": n,
        "missing_before": missing_before,
        "missing_after": missing_after
    }, indent=2))

if __name__ == "__main__":
    main()
