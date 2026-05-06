from pathlib import Path
import re, json, subprocess, platform, os, sys
from datetime import datetime

ROOT = Path.home() / "HypothesisMed"
OUTDIR = ROOT / "paper_assests" / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Search these file types for experiment settings.
SEARCH_EXTS = {".py", ".sh", ".log", ".out", ".txt", ".yaml", ".yml", ".json", ".jsonl"}
MAX_FILE_MB = 80

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"Unavailable ({e})"

def read_text(p):
    try:
        if p.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            return ""
        return p.read_text(errors="ignore")
    except Exception:
        return ""

def collect_files():
    files = []
    skip_dirs = {".git", ".venv", "__pycache__", ".cache"}
    for p in ROOT.rglob("*"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SEARCH_EXTS:
            files.append(p)
    return files

files = collect_files()
all_text = []
for p in files:
    txt = read_text(p)
    if txt:
        all_text.append((p, txt))

def find_values(patterns, flags=re.I):
    hits = []
    for p, txt in all_text:
        for pat in patterns:
            for m in re.finditer(pat, txt, flags):
                val = m.group(1) if m.groups() else m.group(0)
                line_start = txt.rfind("\n", 0, m.start()) + 1
                line_end = txt.find("\n", m.end())
                if line_end == -1:
                    line_end = min(len(txt), m.end() + 200)
                line = txt[line_start:line_end].strip()
                hits.append((str(p.relative_to(ROOT)), val.strip(), line[:260]))
    return hits

def best_value(name, patterns, default=None):
    hits = find_values(patterns)
    # Prefer explicit command/config/log values, avoid README-like noise.
    preferred = []
    for h in hits:
        path, val, line = h
        lower = (path + " " + line).lower()
        if any(x in lower for x in ["run_experiment", "samplingparams", "vllm", "max_new_tokens", "temperature", "top_p", "max_model_len", "dtype", "trust_remote_code"]):
            preferred.append(h)
    use = preferred or hits
    if use:
        # Pick the most frequent value among candidates.
        vals = {}
        for _, val, _ in use:
            val = val.strip().strip("'\"")
            vals[val] = vals.get(val, 0) + 1
        return sorted(vals.items(), key=lambda x: (-x[1], x[0]))[0][0], use[:8]
    return default, []

settings = {}

patterns = {
    "Inference engine": [r"\bvllm\b", r"from\s+vllm\s+import", r"LLM\("],
    "Precision": [r"dtype['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_]+)", r"--dtype\s+([A-Za-z0-9_]+)"],
    "Maximum context length": [r"max_model_len['\"]?\s*[:=]\s*(\d+)", r"--max-model-len\s+(\d+)", r"max_seq_len\s+(\d+)", r"Using max model len\s+(\d+)"],
    "Maximum new tokens": [r"max_new_tokens['\"]?\s*[:=]\s*(\d+)", r"--max-new-tokens\s+(\d+)", r"max_tokens['\"]?\s*[:=]\s*(\d+)"],
    "Temperature": [r"temperature['\"]?\s*[:=]\s*([0-9.]+)", r"--temperature\s+([0-9.]+)"],
    "Top-p": [r"top_p['\"]?\s*[:=]\s*([0-9.]+)", r"--top-p\s+([0-9.]+)"],
    "Seed": [r"seed['\"]?\s*[:=]\s*(\d+)", r"--seed\s+(\d+)"],
    "Batch size": [r"batch_size['\"]?\s*[:=]\s*(\d+)", r"--batch-size\s+(\d+)"],
    "Tensor parallel size": [r"tensor_parallel_size['\"]?\s*[:=]\s*(\d+)", r"--tensor-parallel-size\s+(\d+)"],
    "GPU memory utilization": [r"gpu_memory_utilization['\"]?\s*[:=]\s*([0-9.]+)", r"--gpu-memory-utilization\s+([0-9.]+)"],
}

debug = {}
for k, pats in patterns.items():
    val, hits = best_value(k, pats)
    if k == "Inference engine":
        val = "vLLM" if hits else "Not found"
    if k == "Maximum context length" and val is None:
        val = "4096" if any("max model len 4096" in t.lower() for _, t in all_text) else "Not found"
    settings[k] = val if val is not None else "Not found"
    debug[k] = hits

# Known experimental design from generated results.
settings["Datasets"] = "MedQA, MedMCQA, PubMedQA"
settings["Examples per dataset"] = "1000"
settings["Models"] = "Qwen2.5-7B, Phi-4-mini, DeepSeek-R1-32B, BioMistral-7B"
settings["Prompting methods"] = "Direct, CoT, HypMed-v3"
settings["Fusion rule"] = "Majority vote with parseable fallback"
settings["Gold SPACE label"] = "VALID for original benchmark items"
settings["False-commitment threshold"] = r"confidence $\geq 0.5$"

# Server and software config.
hostname = run("hostname")
kernel = run("uname -sr")
cpu_model = run("lscpu | awk -F: '/Model name/{gsub(/^[ \\t]+/,\"\",$2); print $2; exit}'")
cpu_sockets = run("lscpu | awk -F: '/Socket\\(s\\)/{gsub(/^[ \\t]+/,\"\",$2); print $2; exit}'")
cpu_cores = run("lscpu | awk -F: '/Core\\(s\\) per socket/{gsub(/^[ \\t]+/,\"\",$2); print $2; exit}'")
cpu_threads = run("lscpu | awk -F: '/CPU\\(s\\)/{gsub(/^[ \\t]+/,\"\",$2); print $2; exit}'")
ram = run("free -h | awk '/Mem:/{print $2}'")
gpu_name = run("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
gpu_count = run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
gpu_mem = run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1")
driver = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
cuda_from_smi = run("nvidia-smi | sed -n 's/.*CUDA Version: \\([0-9.]*\\).*/\\1/p' | head -1")
python_ver = run("python -V")
torch_ver = run("python - <<'PY2'\ntry:\n import torch\n print(torch.__version__)\nexcept Exception as e:\n print('Unavailable')\nPY2")
vllm_ver = run("python - <<'PY2'\ntry:\n import vllm\n print(vllm.__version__)\nexcept Exception as e:\n print('Unavailable')\nPY2")
transformers_ver = run("python - <<'PY2'\ntry:\n import transformers\n print(transformers.__version__)\nexcept Exception as e:\n print('Unavailable')\nPY2")

server_settings = {
    "Host": hostname,
    "Operating system / kernel": kernel,
    "CPU": cpu_model,
    "CPU sockets / cores / threads": f"{cpu_sockets} sockets, {cpu_cores} cores/socket, {cpu_threads} threads",
    "System memory": ram,
    "GPU": f"{gpu_count} $\\times$ {gpu_name}",
    "GPU memory": f"{gpu_mem} MiB per GPU",
    "NVIDIA driver": driver,
    "CUDA reported by nvidia-smi": cuda_from_smi,
    "Python": python_ver,
    "PyTorch": torch_ver,
    "vLLM": vllm_ver,
    "Transformers": transformers_ver,
}

def esc(s):
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for a,b in repl.items():
        s = s.replace(a,b)
    return s

# Compact paper table.
paper_rows = [
    ("Inference engine", settings["Inference engine"]),
    ("Precision", settings["Precision"]),
    ("Maximum context length", settings["Maximum context length"] + " tokens" if str(settings["Maximum context length"]).isdigit() else settings["Maximum context length"]),
    ("Maximum new tokens", settings["Maximum new tokens"]),
    ("Temperature", settings["Temperature"]),
    ("Top-p", settings["Top-p"]),
    ("Seed", settings["Seed"]),
    ("Datasets", settings["Datasets"]),
    ("Examples per dataset", settings["Examples per dataset"]),
    ("Prompting methods", settings["Prompting methods"]),
    ("Fusion rule", settings["Fusion rule"]),
    ("Gold SPACE label", settings["Gold SPACE label"]),
    ("False-commitment threshold", settings["False-commitment threshold"]),
    ("Compute node", server_settings["Host"]),
    ("GPU configuration", server_settings["GPU"]),
    ("GPU memory", server_settings["GPU memory"]),
    ("CPU", server_settings["CPU"]),
    ("System memory", server_settings["System memory"]),
]

tex = []
tex.append("% Auto-generated experiment settings table.\n")
tex.append("\\begin{tabular}{ll}\n")
tex.append("\\toprule\n")
tex.append("Setting & Value \\\\\n")
tex.append("\\midrule\n")
for k,v in paper_rows:
    tex.append(f"{esc(k)} & {esc(v)} \\\\\n")
tex.append("\\bottomrule\n")
tex.append("\\end{tabular}\n")

tex_path = OUTDIR / "table_experiment_settings.tex"
tex_path.write_text("".join(tex))

# Full server/software config table for appendix if needed.
server_tex = []
server_tex.append("% Auto-generated server and software configuration table.\n")
server_tex.append("\\begin{tabular}{ll}\n")
server_tex.append("\\toprule\n")
server_tex.append("Item & Value \\\\\n")
server_tex.append("\\midrule\n")
for k,v in server_settings.items():
    server_tex.append(f"{esc(k)} & {esc(v)} \\\\\n")
server_tex.append("\\bottomrule\n")
server_tex.append("\\end{tabular}\n")
server_tex_path = OUTDIR / "table_server_software_config.tex"
server_tex_path.write_text("".join(server_tex))

# Debug evidence file.
debug_path = OUTDIR / "experiment_settings_extraction_debug.txt"
with debug_path.open("w") as f:
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Root: {ROOT}\n")
    f.write(f"Files scanned: {len(files)}\n\n")
    f.write("===== SETTINGS =====\n")
    for k,v in settings.items():
        f.write(f"{k}: {v}\n")
    f.write("\n===== SERVER SETTINGS =====\n")
    for k,v in server_settings.items():
        f.write(f"{k}: {v}\n")
    f.write("\n===== EVIDENCE HITS =====\n")
    for k,hits in debug.items():
        f.write(f"\n--- {k} ---\n")
        if not hits:
            f.write("No hits found.\n")
        for path,val,line in hits:
            f.write(f"[{path}] value={val} | {line}\n")

print("\n===== GENERATED TABLES =====")
print(tex_path)
print(server_tex_path)
print(debug_path)
print("\n===== MAIN EXPERIMENT SETTINGS TABLE =====")
print(tex_path.read_text())
print("\n===== FULL SERVER/SOFTWARE CONFIG TABLE =====")
print(server_tex_path.read_text())
print("\n===== CHECK DEBUG EVIDENCE IF ANY VALUE SAYS Not found =====")
print(f"less {debug_path}")
