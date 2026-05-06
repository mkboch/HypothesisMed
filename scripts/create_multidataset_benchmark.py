import json
from pathlib import Path
from datasets import load_dataset

OUTDIR = Path("datasets/transformed")
OUTDIR.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")

def make_medmcqa(max_n=1000):
    candidates = [
        ("openlifescienceai/medmcqa", None),
        ("medmcqa", None),
    ]

    ds = None
    used = None
    for name, cfg in candidates:
        try:
            print(f"Trying MedMCQA source: {name}")
            ds = load_dataset(name) if cfg is None else load_dataset(name, cfg)
            used = name
            break
        except Exception as e:
            print(f"Failed {name}: {e}")

    if ds is None:
        print("Could not load MedMCQA. Skipping.")
        return

    split = "validation" if "validation" in ds else ("test" if "test" in ds else ("train" if "train" in ds else list(ds.keys())[0]))
    rows = []

    for ex in ds[split]:
        if len(rows) >= max_n:
            break

        q = ex.get("question") or ex.get("Question")
        options = {}

        if all(k in ex for k in ["opa", "opb", "opc", "opd"]):
            options = {
                "A": str(ex["opa"]).strip(),
                "B": str(ex["opb"]).strip(),
                "C": str(ex["opc"]).strip(),
                "D": str(ex["opd"]).strip(),
            }
        elif isinstance(ex.get("options"), dict):
            options = {str(k).strip().upper(): str(v).strip() for k, v in ex["options"].items()}
        elif isinstance(ex.get("Options"), dict):
            options = {str(k).strip().upper(): str(v).strip() for k, v in ex["Options"].items()}

        ans = ex.get("cop", ex.get("answer", ex.get("label", ex.get("Correct Option"))))
        if isinstance(ans, int):
            # MedMCQA cop is usually 0-3
            ans = "ABCD"[ans] if 0 <= ans <= 3 else None
        elif ans is not None:
            ans = str(ans).strip().upper()
            if ans in {"0", "1", "2", "3"}:
                ans = "ABCD"[int(ans)]

        if not q or not options or ans not in options:
            continue

        rows.append({
            "id": f"medmcqa_original1000_{len(rows)}",
            "dataset": "medmcqa",
            "transform": "original",
            "question": str(q).strip(),
            "options": options,
            "gold_answer": ans,
            "gold_space_label": "VALID",
            "answer": ans,
            "space_label": "VALID",
            "source": used,
            "split": split
        })

    write_jsonl(OUTDIR / "medmcqa_original1000.jsonl", rows)

def make_pubmedqa(max_n=1000):
    candidates = [
        ("qiaojin/PubMedQA", "pqa_labeled"),
        ("pubmed_qa", "pqa_labeled"),
    ]

    ds = None
    used = None
    for name, cfg in candidates:
        try:
            print(f"Trying PubMedQA source: {name}, config={cfg}")
            ds = load_dataset(name, cfg)
            used = f"{name}/{cfg}"
            break
        except Exception as e:
            print(f"Failed {name}/{cfg}: {e}")

    if ds is None:
        print("Could not load PubMedQA. Skipping.")
        return

    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = []

    for ex in ds[split]:
        if len(rows) >= max_n:
            break

        q = ex.get("question") or ex.get("QUESTION")
        ans = ex.get("final_decision") or ex.get("answer") or ex.get("label")
        if not q or ans is None:
            continue

        ans = str(ans).strip().lower()
        mapping = {"yes": "A", "no": "B", "maybe": "C"}
        if ans not in mapping:
            continue

        context = ex.get("context", "")
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            if isinstance(contexts, list):
                context_text = " ".join(str(x) for x in contexts[:5])
            else:
                context_text = str(contexts)
        elif isinstance(context, list):
            context_text = " ".join(str(x) for x in context[:5])
        else:
            context_text = str(context)

        question_text = (
            "Abstract/context:\n"
            + context_text[:3000]
            + "\n\nQuestion:\n"
            + str(q).strip()
            + "\n\nBased on the abstract, what is the answer?"
        )

        rows.append({
            "id": f"pubmedqa_original1000_{len(rows)}",
            "dataset": "pubmedqa",
            "transform": "original",
            "question": question_text,
            "options": {"A": "yes", "B": "no", "C": "maybe"},
            "gold_answer": mapping[ans],
            "gold_space_label": "VALID",
            "answer": mapping[ans],
            "space_label": "VALID",
            "source": used,
            "split": split
        })

    write_jsonl(OUTDIR / "pubmedqa_original1000.jsonl", rows)

make_medmcqa(1000)
make_pubmedqa(1000)

print("\nFinal transformed files:")
for p in sorted(OUTDIR.glob("*original1000.jsonl")):
    print(p, sum(1 for _ in open(p)))
