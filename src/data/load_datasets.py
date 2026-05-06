import random
from datasets import load_dataset
from src.utils.io import write_jsonl

def load_medmcqa(max_samples=500, seed=42):
    ds = load_dataset("openlifescienceai/medmcqa")
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    rows = []
    for i, ex in enumerate(ds[split]):
        q = ex.get("question", "")
        options = {
            "A": str(ex.get("opa", "")),
            "B": str(ex.get("opb", "")),
            "C": str(ex.get("opc", "")),
            "D": str(ex.get("opd", ""))
        }
        cop = ex.get("cop")
        if cop is None:
            continue
        ans = ["A", "B", "C", "D"][int(cop)]
        rows.append({
            "id": f"medmcqa_{i}",
            "dataset": "medmcqa",
            "question": q,
            "options": options,
            "answer": ans
        })
    random.Random(seed).shuffle(rows)
    return rows[:max_samples]

def main():
    rows = load_medmcqa()
    write_jsonl("datasets/processed/original.jsonl", rows)
    print(f"Saved {len(rows)} examples.")

if __name__ == "__main__":
    main()
