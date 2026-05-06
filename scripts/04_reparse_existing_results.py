import argparse
import json
from pathlib import Path

from src.evaluation.parse import extract_json, extract_answer, extract_space_label, extract_confidence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open() as f, out.open("w") as g:
        for line in f:
            d = json.loads(line)
            text = d.get("raw_output", "")
            obj = extract_json(text)

            d["parsed_output"] = obj
            d["pred_answer"] = extract_answer(obj, text)
            d["pred_space_label"] = extract_space_label(obj, text)
            d["confidence"] = extract_confidence(obj, text)

            g.write(json.dumps(d) + "\n")
            n += 1

    print(f"Reparsed {n} rows")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
