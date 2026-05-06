import os
import json
from pathlib import Path

roots = []
for env in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"]:
    v = os.environ.get(env)
    if v:
        roots.append(Path(v))

roots += [
    Path.home() / ".cache/huggingface/hub",
    Path("/home/manikm/lab_chatbot_h100/cache/hf/hub"),
    Path("/scratch/manikm/cache/hf/hub"),
    Path("/scratch/manikm/hf/hub"),
    Path("/scratch/manikm/.cache/huggingface/hub"),
]

seen = set()
models = []

bad_terms = [
    "vl", "vision", "ocr", "clip", "siglip", "whisper", "wav", "audio",
    "embed", "embedding", "reranker", "bge", "e5", "bert", "roberta",
    "deberta", "layout", "sam", "diffusion", "sdxl"
]

priority_terms = [
    "qwen3",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "gemma-2-9b",
    "llama-3.1-8b",
    "llama-3-8b",
    "mistral-7b",
    "mixtral",
]

exclude_terms = [
    "qwen2.5-7b-instruct",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-ocr",
    "qwen3-vl",
]

for root in roots:
    if not root.exists():
        continue

    for p in root.glob("models--*"):
        name = p.name.replace("models--", "").replace("--", "/")
        lname = name.lower()

        if name in seen:
            continue
        seen.add(name)

        if any(x in lname for x in bad_terms):
            continue
        if any(x in lname for x in exclude_terms):
            continue

        snapshots = p / "snapshots"
        if not snapshots.exists():
            continue

        snap_dirs = [s for s in snapshots.iterdir() if s.is_dir()]
        if not snap_dirs:
            continue

        # choose newest snapshot by modification time
        snap = max(snap_dirs, key=lambda x: x.stat().st_mtime)

        has_config = (snap / "config.json").exists()
        has_tokenizer = (
            (snap / "tokenizer.json").exists()
            or (snap / "tokenizer.model").exists()
            or (snap / "vocab.json").exists()
        )
        has_weights = bool(list(snap.glob("*.safetensors"))) or bool(list(snap.glob("*.bin")))

        if not (has_config and has_tokenizer and has_weights):
            continue

        score = 0
        for i, term in enumerate(priority_terms):
            if term in lname:
                score += 100 - i

        # prefer instruct/chat models
        if "instruct" in lname or "it" in lname or "chat" in lname:
            score += 20

        models.append({
            "name": name,
            "local_path": str(snap),
            "cache_root": str(root),
            "score": score,
            "mtime": snap.stat().st_mtime
        })

models = sorted(models, key=lambda x: (x["score"], x["mtime"]), reverse=True)

Path("results/local_extra_models").mkdir(parents=True, exist_ok=True)
with open("results/local_extra_models/cached_text_model_inventory.json", "w") as f:
    json.dump(models, f, indent=2)

selected = models[:3]
with open("results/local_extra_models/selected_extra_models.json", "w") as f:
    json.dump(selected, f, indent=2)

print("===== Cached text model candidates =====")
for m in models[:20]:
    print(f'{m["score"]:4d}  {m["name"]}  {m["local_path"]}')

print("\n===== Selected extra models =====")
for m in selected:
    print(f'{m["name"]}  {m["local_path"]}')
