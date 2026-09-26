#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList, set_seed


ROOT = Path(os.environ.get("HYPOTHESISMED_ROOT", Path(__file__).resolve().parents[2]))
PROTOCOL = ROOT / "results/final_self_consistency_protocol_freeze_20260831"
OUT_DIR = ROOT / "results/final_self_consistency_generations_20260831"
INPUTS = PROTOCOL / "sc3_cot_inputs_native_chat.jsonl"
SETTINGS = PROTOCOL / "sc3_generation_settings.json"
MAX_MODEL_LEN = 8192
MAX_TOKENS = 4096
CONDITION = "native_chat"
FSYNC_EVERY = 10


class PresencePenalty(LogitsProcessor):
    def __init__(self, penalty: float, prompt_lengths: list[int]):
        self.penalty = float(penalty)
        self.prompt_lengths = prompt_lengths

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.penalty == 0:
            return scores
        for row_idx in range(input_ids.shape[0]):
            generated = input_ids[row_idx, self.prompt_lengths[row_idx] :]
            if generated.numel():
                scores[row_idx, torch.unique(generated)] -= self.penalty
        return scores


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def severe_repetition(text: str) -> bool:
    words = re.findall(r"\S+", (text or "").lower())
    if len(words) < 4:
        return False
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = Counter(grams)
    return (1.0 - len(counts) / len(grams)) >= 0.50 or max(counts.values()) >= 20


def reasoning_marker(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in ["<think>", "</think>", "<|think|>", "<|channel>thought", "<|channel|>thought", "<unused94>thought", "<thought>", "</thought>"])


def render_native(tokenizer, prompt: str, kwargs: dict) -> str:
    k = {"tokenize": False, "add_generation_prompt": True}
    k.update(kwargs)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **k)
    if not isinstance(rendered, str):
        raise RuntimeError("chat template did not return string")
    return rendered


def load_completed(path: Path):
    done = set()
    bad = 0
    if not path.exists():
        return done, bad
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            done.add(tuple(r["unique_key"]))
    return done, bad


def generation_kwargs(cfg, eos_token_id, pad_token_id, prompt_length):
    d = cfg["sc_decoding"]
    kw = {
        "max_new_tokens": MAX_TOKENS,
        "do_sample": bool(d["do_sample"]),
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "repetition_penalty": d["repetition_penalty"],
    }
    if d["do_sample"]:
        kw.update(temperature=d["temperature"], top_p=d["top_p"])
        if d.get("top_k") is not None:
            kw["top_k"] = d["top_k"]
    if d.get("presence_penalty"):
        kw["logits_processor"] = LogitsProcessorList([PresencePenalty(d["presence_penalty"], [prompt_length])])
    return kw


def model_settings():
    rows = json.loads(SETTINGS.read_text(encoding="utf-8"))["models"]
    return {r["model_key"]: r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(model_settings()))
    args = ap.parse_args()
    cfg = model_settings()[args.model_key]
    seeds = list(cfg["sc_seeds"])
    if seeds != [20260831, 20260832, 20260833]:
        raise RuntimeError(f"Unexpected SC seeds: {seeds}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6,7":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 6,7")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.model_key}_sc3_cot_samples.jsonl"
    manifest_path = OUT_DIR / f"{args.model_key}_sc3_completion_manifest.json"
    log_path = OUT_DIR / f"{args.model_key}_sc3_run_status.json"
    inputs = load_jsonl(INPUTS)
    if len(inputs) != 5956:
        raise RuntimeError(f"Expected 5956 SC inputs, found {len(inputs)}")
    done, bad = load_completed(output_path)
    if bad:
        raise RuntimeError(f"{output_path} has {bad} corrupted JSONL rows; stop before resume")

    pending = []
    for row in inputs:
        for sample_idx, seed in enumerate(seeds, start=1):
            sample_id = f"sample_{sample_idx}"
            key = (cfg["model_id"], cfg["revision"], CONDITION, row["dataset"], int(row["source_index"]), row["uid"], sample_id, int(seed))
            if key not in done:
                pending.append((row, sample_id, int(seed)))
    total = len(inputs) * len(seeds)
    print(f"MODEL={cfg['model_id']} REVISION={cfg['revision']} TOTAL={total} DONE={len(done)} PENDING={len(pending)}", flush=True)
    log_path.write_text(json.dumps({"status": "loading", "model_key": args.model_key, "done": len(done), "pending": len(pending), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n", encoding="utf-8")
    if not pending:
        print("NO_PENDING_ROWS", flush=True)
        return 0

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], local_files_only=False, trust_remote_code=cfg["tokenizer_trust_remote_code"])
    model_cls = AutoModelForImageTextToText if cfg["model_class"] == "AutoModelForImageTextToText" else AutoModelForCausalLM
    model = model_cls.from_pretrained(cfg["model_id"], revision=cfg["revision"], dtype=torch.bfloat16, local_files_only=False, trust_remote_code=cfg["model_trust_remote_code"])
    model.eval()
    model.to("cuda:0")
    eos_token_id = model.generation_config.eos_token_id or getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id

    generated = 0
    with output_path.open("a", encoding="utf-8") as f:
        for idx, (row, sample_id, seed) in enumerate(pending, start=1):
            set_seed(seed)
            serialized = render_native(tokenizer, row["prompt"], cfg["chat_template_kwargs"])
            enc = tokenizer(serialized, return_tensors="pt", return_dict=True)
            prompt_tokens = int(enc["input_ids"].shape[-1])
            if prompt_tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError(f"context overflow {row['dataset']} {row['source_index']} {sample_id} {prompt_tokens}+{MAX_TOKENS}>{MAX_MODEL_LEN}")
            enc = enc.to("cuda:0")
            with torch.inference_mode():
                out = model.generate(**enc, **generation_kwargs(cfg, eos_token_id, pad_token_id, prompt_tokens))
            gen_ids = out[0, enc["input_ids"].shape[-1] :]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            diag = tokenizer.decode(gen_ids, skip_special_tokens=False)
            out_tokens = int(gen_ids.numel())
            rec = {
                "unique_key": [cfg["model_id"], cfg["revision"], CONDITION, row["dataset"], int(row["source_index"]), row["uid"], sample_id, seed],
                "model_key": args.model_key,
                "model_id": cfg["model_id"],
                "revision": cfg["revision"],
                "backend": cfg["backend"],
                "runtime_environment": cfg["runtime_environment"],
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_build": torch.version.cuda,
                "transformers_version": transformers.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "condition": CONDITION,
                "method": "cot",
                "sc_method": "SC-3",
                "sample_id": sample_id,
                "sample_index": int(sample_id.split("_")[1]),
                "dataset": row["dataset"],
                "source_index": int(row["source_index"]),
                "uid": row["uid"],
                "gold": row.get("gold"),
                "seed": seed,
                "decoding_settings": cfg["sc_decoding"],
                "sc_decoding_source": cfg["sc_decoding_source"],
                "reasoning_policy": cfg["reasoning_policy"],
                "reasoning_control": cfg["reasoning_control"],
                "max_model_len": MAX_MODEL_LEN,
                "max_tokens": MAX_TOKENS,
                "controlled_prompt": row["prompt"],
                "controlled_prompt_sha256": row["controlled_prompt_sha256"],
                "serialized_input": serialized,
                "serialized_input_sha256": sha_text(serialized),
                "prompt_tokens": prompt_tokens,
                "raw_output": raw,
                "diagnostic_decode_with_special_tokens": diag,
                "output_tokens": out_tokens,
                "finish_reason": "length" if out_tokens >= MAX_TOKENS else "eos_or_stop",
                "empty_completion": not bool(raw.strip()),
                "length_truncation": out_tokens >= MAX_TOKENS,
                "severe_repetition": severe_repetition(raw),
                "reasoning_marker": reasoning_marker(diag),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1
            if generated % FSYNC_EVERY == 0:
                os.fsync(f.fileno())
            if idx == 1 or idx % 25 == 0:
                print(f"row={idx}/{len(pending)} dataset={row['dataset']} sample={sample_id} out={out_tokens} finish={rec['finish_reason']} empty={rec['empty_completion']} trunc={rec['length_truncation']} rep={rec['severe_repetition']} marker={rec['reasoning_marker']}", flush=True)
                log_path.write_text(json.dumps({"status": "running", "model_key": args.model_key, "generated_this_run": generated, "pending_at_start": len(pending), "last_row": idx, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n", encoding="utf-8")
        os.fsync(f.fileno())

    records = load_jsonl(output_path)
    keys = [tuple(r["unique_key"]) for r in records]
    manifest = {
        "model_key": args.model_key,
        "model_id": cfg["model_id"],
        "revision": cfg["revision"],
        "condition": CONDITION,
        "expected_rows": total,
        "rows": len(records),
        "unique_keys": len(set(keys)),
        "duplicate_keys": len(keys) - len(set(keys)),
        "empty_completions": sum(r["empty_completion"] for r in records),
        "length_truncations": sum(r["length_truncation"] for r in records),
        "severe_repetition_rows": sum(r["severe_repetition"] for r in records),
        "reasoning_marker_rows": sum(r["reasoning_marker"] for r in records),
        "finish_reason_counts": dict(Counter(r["finish_reason"] for r in records)),
        "sample_rows": dict(Counter(r["sample_id"] for r in records)),
        "output_jsonl": str(output_path.relative_to(ROOT)),
        "output_sha256": sha_file(output_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete": len(records) == total and len(set(keys)) == total,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(json.dumps({"status": "complete", "manifest": str(manifest_path.relative_to(ROOT)), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
