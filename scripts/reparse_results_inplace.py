import argparse
import json
import shutil
from pathlib import Path
from src.evaluation.parser import parse_output

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
args = ap.parse_args()

for path in sorted(Path(".").glob(args.glob)):
    if not path.exists() or path.stat().st_size == 0:
        continue

    backup = path.with_suffix(path.suffix + ".before_multidata_reparse")
    if not backup.exists():
        shutil.copy2(path, backup)

    rows = []
    before_missing = 0
    after_missing = 0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("pred_answer") is None:
                before_missing += 1

            parsed = parse_output(r.get("raw_output", ""))
            if parsed.get("answer") is not None:
                r["pred_answer"] = parsed["answer"]
            if parsed.get("space_label") is not None:
                r["pred_space_label"] = parsed["space_label"]
            if parsed.get("confidence") is not None:
                r["confidence"] = parsed["confidence"]

            if r.get("pred_answer") is None:
                after_missing += 1
            rows.append(r)

    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "file": str(path),
        "n": len(rows),
        "missing_before": before_missing,
        "missing_after": after_missing
    }, indent=2))
