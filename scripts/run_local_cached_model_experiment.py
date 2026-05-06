import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.methods.prompts import build_prompt
from src.evaluation.parser import parse_output

def safe_key(name):
    name = name.lower()
    name = name.replace("/", "_").replace("-", "_").replace(".", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--method", required=True, choices=["direct", "cot", "hypothesismed_v3"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    model_key = safe_key(args.model_name)
    dataset_stem = Path(args.data).stem
    out_path = Path("results") / f"{model_key}_{args.method}_{dataset_stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and sum(1 for _ in out_path.open()) >= args.max_samples:
        print(f"[SKIP] Existing complete file: {out_path}")
        return

    rows = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.max_samples:
                break

    print(f"[INFO] model_name={args.model_name}")
    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] method={args.method}")
    print(f"[INFO] data={args.data}")
    print(f"[INFO] output={out_path}")
    print(f"[INFO] n={len(rows)}")

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        disable_log_stats=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    with out_path.open("w") as out:
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i+args.batch_size]
            prompts = [build_prompt(args.method, r) for r in batch]
            outputs = llm.generate(prompts, sampling)

            for r, o in zip(batch, outputs):
                raw = o.outputs[0].text if o.outputs else ""
                parsed = parse_output(raw)

                result = dict(r)
                result["model"] = model_key
                result["model_name"] = args.model_name
                result["model_path"] = args.model_path
                result["method"] = args.method
                result["raw_output"] = raw
                result["parsed_output"] = parsed
                result["pred_answer"] = parsed.get("answer")
                result["pred_space_label"] = parsed.get("space_label")
                result["confidence"] = parsed.get("confidence", 0.0)

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
