import argparse
import yaml
from tqdm import tqdm
from src.utils.io import read_jsonl, write_jsonl
from src.methods.prompts import build_prompt
from src.models.vllm_runner import VLLMRunner
from src.evaluation.parse import extract_json, extract_answer, extract_space_label, extract_confidence

def get_model_config(model_name):
    cfg = yaml.safe_load(open("configs/models.yaml", "r", encoding="utf-8"))
    for m in cfg["models"]:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", default="datasets/transformed/hypothesismed_eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    model_cfg = get_model_config(args.model)
    rows = read_jsonl(args.data)[:args.max_samples]

    from pathlib import Path as _Path
    _prompt_file = _Path(f"prompts/{args.method}.txt")
    if not _prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found for method={args.method}: {_prompt_file}")


    runner = VLLMRunner(
        model_id=model_cfg["hf_id"],
        tensor_parallel_size=model_cfg["tensor_parallel_size"],
        max_model_len=args.max_model_len
    )

    outputs = []

    for i in tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[i:i + args.batch_size]
        prompts = [build_prompt(args.method, row) for row in batch]
        texts = runner.generate(prompts)

        for row, text in zip(batch, texts):
            obj = extract_json(text)
            outputs.append({
                "id": row["id"],
                "dataset": row["dataset"],
                "transform": row["transform"],
                "method": args.method,
                "model": args.model,
                "question": row["question"],
                "options": row["options"],
                "gold_answer": row["answer"],
                "gold_space_label": row["space_label"],
                "raw_output": text,
                "parsed_output": obj,
                "pred_answer": extract_answer(obj, text),
                "pred_space_label": extract_space_label(obj, text),
                "confidence": extract_confidence(obj, text)
            })

    from pathlib import Path as _Path
    data_stem = _Path(args.data).stem
    out_path = f"results/{args.model}_{args.method}_{data_stem}.jsonl"
    write_jsonl(out_path, outputs)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
