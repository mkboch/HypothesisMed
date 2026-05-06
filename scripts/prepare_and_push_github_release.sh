#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/HypothesisMed"
REPO_URL="https://github.com/mkboch/HypothesisMed.git"
WORK="$HOME/HypothesisMed_github_release"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "0. Safety checks"
echo "============================================================"
cd "$PROJECT"
pwd
git --version || true

echo
echo "Checking GitHub authentication/remote access..."
if ! git ls-remote "$REPO_URL" >/dev/null 2>&1; then
  echo "ERROR: Cannot access $REPO_URL from this server."
  echo "Fix GitHub authentication first. For HTTPS, use a GitHub personal access token when prompted."
  echo "Or configure SSH and change REPO_URL to git@github.com:mkboch/HypothesisMed.git in this script."
  exit 1
fi

echo
echo "============================================================"
echo "1. Recreate clean local release folder"
echo "============================================================"
rm -rf "$WORK"
git clone "$REPO_URL" "$WORK"
cd "$WORK"

echo
echo "Current remote:"
git remote -v

echo
echo "============================================================"
echo "2. Clean repository content except .git"
echo "============================================================"
find . -mindepth 1 -maxdepth 1 ! -name ".git" -exec rm -rf {} +

mkdir -p \
  src \
  experiments \
  scripts \
  results/final_4model_paper \
  results/final_png_only_assets/figures \
  results/final_png_only_assets/tables \
  results/final_claim_validation \
  paper_assests/figures \
  paper_assests/tables \
  reproducibility

echo
echo "============================================================"
echo "3. Copy source code and experiment scripts"
echo "============================================================"

# Core source code
if [ -d "$PROJECT/src" ]; then
  rsync -av "$PROJECT/src/" src/
fi

# Experiment runners
if [ -d "$PROJECT/experiments" ]; then
  rsync -av \
    --include='*/' \
    --include='*.py' \
    --include='*.sh' \
    --include='*.yaml' \
    --include='*.yml' \
    --exclude='*' \
    "$PROJECT/experiments/" experiments/
fi

# Scripts needed for running, reparsing, summarizing, figure/table generation.
if [ -d "$PROJECT/scripts" ]; then
  rsync -av \
    --include='*/' \
    --include='*.py' \
    --include='*.sh' \
    --exclude='*.log' \
    --exclude='__pycache__/' \
    --exclude='*' \
    "$PROJECT/scripts/" scripts/
fi

echo
echo "============================================================"
echo "4. Copy final tables, figures, and claim-validation files"
echo "============================================================"

# Main final result tables
if [ -d "$PROJECT/results/final_4model_paper" ]; then
  rsync -av \
    --include='*.csv' \
    --include='*.tex' \
    --include='*.txt' \
    --exclude='*' \
    "$PROJECT/results/final_4model_paper/" results/final_4model_paper/
fi

# PNG-only paper assets
if [ -d "$PROJECT/results/final_png_only_assets/figures" ]; then
  rsync -av "$PROJECT/results/final_png_only_assets/figures/" results/final_png_only_assets/figures/
  rsync -av "$PROJECT/results/final_png_only_assets/figures/" paper_assests/figures/
fi

if [ -d "$PROJECT/results/final_png_only_assets/tables" ]; then
  rsync -av \
    --include='*.csv' \
    --include='*.tex' \
    --exclude='confidence_rows_from_fusion_outputs.csv' \
    --exclude='confidence_rows_from_fusion_outputs_longtable.tex' \
    --exclude='*' \
    "$PROJECT/results/final_png_only_assets/tables/" results/final_png_only_assets/tables/

  rsync -av \
    --include='*.csv' \
    --include='*.tex' \
    --exclude='confidence_rows_from_fusion_outputs.csv' \
    --exclude='confidence_rows_from_fusion_outputs_longtable.tex' \
    --exclude='*' \
    "$PROJECT/results/final_png_only_assets/tables/" paper_assests/tables/
fi

# Literature-positioning / claim validation
if [ -d "$PROJECT/results/final_claim_validation" ]; then
  rsync -av \
    --include='*.csv' \
    --include='*.txt' \
    --include='*.tex' \
    --exclude='*' \
    "$PROJECT/results/final_claim_validation/" results/final_claim_validation/
fi

echo
echo "============================================================"
echo "5. Create reproducibility metadata"
echo "============================================================"

cat > reproducibility/inference_and_compute_settings.txt <<'EOF'
Inference engine: vLLM
Precision: bfloat16
Maximum context length: 4096
Maximum generation length: 512 tokens for final cached-model evaluation scripts
Temperature: 0.0
Top-p: 1.0
Seed: 42
Datasets: MedQA, MedMCQA, PubMedQA
Examples per dataset: 1000
Prompting methods: Direct, chain-of-thought, HypothesisMed-v3, proposed fusion
Fusion rule: majority vote with deterministic parseable fallback order: chain-of-thought, direct prompting, HypothesisMed-v3
Gold SPACE label for original benchmark items: VALID
False commitment threshold: confidence >= 0.5

Server:
Host: axis2
OS: Ubuntu 22.04.5 LTS
Kernel: Linux 5.15.0-152-generic
CPU: Intel Xeon Platinum 8468, 2 sockets, 48 cores/socket, 192 threads total
RAM: 1.0 TiB
GPU: 8 x NVIDIA H100 80GB HBM3
NVIDIA driver: 550.144.03
CUDA from nvidia-smi: 12.4

Python/package versions from final environment:
Python: 3.13.11
PyTorch: 2.10.0+cu128
vLLM: 0.19.1
Transformers: 5.6.2
NumPy: 2.2.6
Pandas: 3.0.2
EOF

cat > reproducibility/model_snapshots.txt <<'EOF'
Qwen/Qwen2.5-7B-Instruct:
snapshot a09a35458c702b33eeacc393d103063234e8bc28

microsoft/Phi-4-mini-instruct:
snapshot cfbefacb99257ffa30c83adab238a50856ac3083

deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:
snapshot 711ad2ea6aa40cfca18895e8aca02ab92df1a746

BioMistral/BioMistral-7B:
snapshot 9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5
EOF

python - <<'PY' > reproducibility/python_environment_freeze.txt
import sys, subprocess
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\n===== pip freeze =====")
subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False)
PY

cat > reproducibility/data_policy.txt <<'EOF'
This repository intentionally does not include raw MedQA, MedMCQA, or PubMedQA benchmark examples, nor raw model-output JSONL files that may contain benchmark question text. The repository includes source code, prompts, scripts, aggregate results, final tables, figures, and reproducibility metadata. Users should obtain the benchmark datasets from their official sources and run the scripts locally.
EOF

cat > reproducibility/artifact_manifest.txt <<EOF
Created: $STAMP

Included:
- src/: core prompt/model/data/evaluation code
- experiments/: experiment entry points
- scripts/: run, reparse, summarize, table, and figure generation scripts
- results/final_4model_paper/: final aggregate tables and deltas
- results/final_png_only_assets/figures/: final PNG figures used in the paper
- results/final_png_only_assets/tables/: final compact CSV/TeX tables used in the paper
- results/final_claim_validation/: literature-positioning CSV/TXT files
- paper_assests/: copy of final paper figure/table assets for direct Overleaf use
- reproducibility/: inference settings, server configuration, model snapshots, environment freeze, and data policy

Excluded:
- raw datasets
- raw result JSONL files
- large confidence row dumps
- cache files
- model weights
EOF

echo
echo "============================================================"
echo "6. Add .gitignore"
echo "============================================================"

cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.log
*.tmp
*.bak
.DS_Store
.cache/
.venv/
venv/
env/
datasets/raw/
datasets/downloaded/
results/**/*.jsonl
results/**/confidence_rows_from_fusion_outputs.csv
results/**/confidence_rows_from_fusion_outputs_longtable.tex
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
EOF

echo
echo "============================================================"
echo "7. Show final repository tree"
echo "============================================================"
find . -maxdepth 3 -type f | sort | sed 's#^\./##'

echo
echo "============================================================"
echo "8. Check no raw JSONL or raw dataset files are included"
echo "============================================================"
if find . -type f \( -name "*.jsonl" -o -path "./datasets/raw/*" -o -path "./datasets/downloaded/*" \) | grep -q .; then
  echo "ERROR: raw JSONL or raw dataset files found. Aborting."
  find . -type f \( -name "*.jsonl" -o -path "./datasets/raw/*" -o -path "./datasets/downloaded/*" \)
  exit 1
else
  echo "OK: no raw JSONL or raw dataset files included."
fi

echo
echo "============================================================"
echo "9. Commit and push"
echo "============================================================"
git add .
git status --short

if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Add HypothesisMed reproducibility artifacts"
fi

git branch --show-current
git push origin HEAD

echo
echo "DONE. Uploaded reproducibility artifacts to:"
echo "$REPO_URL"
