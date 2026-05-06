import argparse
import json
from collections import Counter

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    rows = list(read_jsonl(args.path))
    print(f"rows={len(rows)}")
    print("pred_answer counts:", Counter(r.get("pred_answer") for r in rows))
    print("gold_answer counts:", Counter(r.get("gold_answer") for r in rows))
    print("pred_space_label counts:", Counter(r.get("pred_space_label") for r in rows))
    print("gold_space_label counts:", Counter(r.get("gold_space_label") for r in rows))

    bad_parse = [r for r in rows if r.get("pred_answer") is None]
    print(f"missing pred_answer={len(bad_parse)}")

    print("\nSample outputs:")
    for r in rows[:args.n]:
        print("=" * 80)
        print("ID:", r.get("id"))
        print("Q:", r.get("question"))
        print("Options:", r.get("options"))
        print("Gold:", r.get("gold_answer"), "Pred:", r.get("pred_answer"))
        print("Gold space:", r.get("gold_space_label"), "Pred space:", r.get("pred_space_label"))
        print("Confidence:", r.get("confidence"))
        print("Raw tail:")
        print((r.get("raw_output") or "")[-1200:])

if __name__ == "__main__":
    main()
