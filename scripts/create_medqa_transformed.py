import json
import random
import sys
import os
from pathlib import Path

ROOT = os.getcwd()

# Avoid local ./datasets directory shadowing Hugging Face datasets package
sys.path = [p for p in sys.path if p not in ("", ROOT)]

from datasets import load_dataset

OUT = Path(ROOT) / "datasets/transformed/medqa_hypothesismed_eval.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

def norm_options(raw):
    if isinstance(raw, dict):
        return {str(k).strip().upper(): str(v).strip() for k, v in raw.items()}
    if isinstance(raw, list):
        letters = "ABCDE"
        return {letters[i]: str(v).strip() for i, v in enumerate(raw)}
    return {}

def get_field(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None

ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf")
split = "test" if "test" in ds else list(ds.keys())[0]

rows = []

for i, r in enumerate(ds[split]):
    q = get_field(r, ["question", "Question"])
    opts = norm_options(get_field(r, ["options", "Options", "choices"]))

    ans = get_field(r, ["answer_idx", "answer", "label"])
    if isinstance(ans, int):
        gold = "ABCD"[ans]
    else:
        ans = str(ans).strip().upper()
        gold = ans[0] if ans else None

    if not q or not opts or gold not in opts:
        continue

    rows.append({
        "id": f"medqa_{i}_valid",
        "dataset": "medqa",
        "transform": "original",
        "question": q,
        "options": opts,
        "gold_answer": gold,
        "gold_space_label": "VALID"
    })

    inc_opts = {k: v for k, v in opts.items() if k != gold}
    inc_opts["E"] = "None of the above"

    rows.append({
        "id": f"medqa_{i}_incomplete",
        "dataset": "medqa",
        "transform": "incomplete",
        "question": q,
        "options": inc_opts,
        "gold_answer": "E",
        "gold_space_label": "INCOMPLETE"
    })

    wrong_keys = [k for k in opts if k != gold]
    if wrong_keys:
        bad = random.choice(wrong_keys)
        contra_opts = dict(opts)
        contra_opts[gold] = contra_opts[bad]
        contra_opts["E"] = "None of the above"

        rows.append({
            "id": f"medqa_{i}_contradicted",
            "dataset": "medqa",
            "transform": "contradicted",
            "question": q,
            "options": contra_opts,
            "gold_answer": "E",
            "gold_space_label": "CONTRADICTED"
        })

rows = rows[:1000]

with OUT.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"Saved {OUT} with {len(rows)} rows")
