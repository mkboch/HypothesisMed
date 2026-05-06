import json
from pathlib import Path
from datasets import load_dataset

OUT = Path("datasets/transformed/medqa_original1000.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("openlifescienceai/medqa")
split = "test" if "test" in ds else list(ds.keys())[0]

rows = []
for i, ex in enumerate(ds[split]):
    if len(rows) >= 1000:
        break

    d = ex.get("data", ex)

    question = d.get("Question") or d.get("question") or d.get("sent1")
    options = d.get("Options") or d.get("options")

    if options is None:
        opts = {}
        for j, letter in enumerate("ABCDE"):
            key = f"ending{j}"
            if key in d:
                opts[letter] = str(d[key]).strip()
        options = opts

    if isinstance(options, list):
        options = {letter: str(options[j]).strip() for j, letter in enumerate("ABCDE") if j < len(options)}
    elif isinstance(options, dict):
        options = {str(k).strip().upper(): str(v).strip() for k, v in options.items()}
    else:
        continue

    answer = (
        d.get("Correct Option")
        or d.get("correct_option")
        or d.get("answer")
        or d.get("label")
        or ex.get("label")
    )

    if isinstance(answer, int):
        answer = "ABCDE"[answer]
    else:
        answer = str(answer).strip().upper()
        if answer not in options:
            # Sometimes answer is answer text
            for k, v in options.items():
                if answer.lower() == str(v).strip().lower():
                    answer = k
                    break

    if not question or answer not in options:
        continue

    rows.append({
        "id": f"medqa_original1000_{len(rows)}",
        "dataset": "medqa",
        "transform": "original",
        "question": question,
        "options": options,
        "gold_answer": answer,
        "gold_space_label": "VALID",
        "answer": answer,
        "space_label": "VALID"
    })

with OUT.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows to {OUT}")
if len(rows) < 1000:
    raise RuntimeError(f"Only created {len(rows)} rows")
