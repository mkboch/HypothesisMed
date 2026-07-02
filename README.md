# HypothesisMed

**Inference-time answer fusion and structured hypothesis-space reporting for biomedical question answering**

HypothesisMed is a reproducibility repository for evaluating biomedical multiple-choice question answering systems beyond final-answer accuracy. The code supports inference-time prompting, answer parsing, answer fusion, structured SPACE-label reporting, and reliability-oriented analysis across biomedical QA benchmarks.

This repository accompanies the manuscript:

> **HypothesisMed: Inference-Time Answer Fusion and Structured Hypothesis-Space Reporting for Biomedical Question Answering**

## Overview

Biomedical question answering models are often evaluated mainly by final-answer accuracy. HypothesisMed separates final-answer selection from additional reliability signals, including parseability, structured output compliance, answer-space status, confidence reporting, and false-commitment behavior.

The repository provides code for direct prompting, chain-of-thought prompting, structured HypothesisMed prompting, answer fusion, SPACE-label reporting, parse and coverage analysis, reliability and audit summaries, and expanded local-model, medical-model, Claude, and MedGemma evaluations.

The goal is not to claim that one prompting strategy universally dominates all baselines. The goal is to provide a structured and reproducible workflow for studying when inference-time reporting and fusion help, when they fail, and how reliably models expose answer-space problems.

## Repository structure

    .
    ├── configs/
    │   └── models.yaml
    ├── experiments/
    │   ├── __init__.py
    │   ├── audit_results.py
    │   ├── evaluate_results.py
    │   ├── reliability_analysis.py
    │   └── run_experiment.py
    ├── scripts/
    │   ├── analysis/
    │   ├── claude/
    │   ├── data/
    │   ├── expanded_evaluation/
    │   ├── medgemma/
    │   ├── original/
    │   ├── scaled/
    │   └── setup/
    ├── src/
    │   ├── evaluation/
    │   ├── methods/
    │   └── utils/
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## Core components

### Source code

- src/methods/prompts.py  
  Prompt templates and method definitions.

- src/evaluation/parser.py and src/evaluation/parse.py  
  Output parsing utilities for extracting answers and structured fields.

- src/evaluation/metrics.py  
  Accuracy, coverage, and reliability metrics.

- src/utils/io.py  
  Input and output helpers.

### Experiment entry points

- experiments/run_experiment.py  
  Main experiment launcher.

- experiments/evaluate_results.py  
  Evaluation and summary utilities.

- experiments/audit_results.py  
  Structured audit utilities.

- experiments/reliability_analysis.py  
  Reliability-oriented post-processing.

### Reproduction scripts

- scripts/setup/  
  Environment setup, data preparation, and smoke tests.

- scripts/data/  
  Dataset construction, benchmark preparation, and fusion-input generation.

- scripts/original/  
  Main original-model run script.

- scripts/scaled/  
  Scaled evaluation scripts.

- scripts/expanded_evaluation/  
  Additional model, ablation, and expanded evaluation scripts.

- scripts/claude/  
  Claude batch and tool-enforced structured-output experiments.

- scripts/medgemma/  
  MedGemma download and evaluation scripts.

- scripts/analysis/  
  Final checks, reparsing, diagnostics, and reliability summaries.

## SPACE labels

HypothesisMed uses SPACE labels to describe the status of the answer space.

| Label | Meaning |
|---|---|
| VALID | The answer options contain a supported best answer. |
| INCOMPLETE | The answer options appear insufficient or may omit the best answer. |
| CONTRADICTED | The answer options appear internally inconsistent, duplicated, ambiguous, or contradictory. |

SPACE labels are structured reporting signals. They should be evaluated separately from final-answer accuracy.

## Installation

Create a clean Python environment and install the required packages.

Linux or macOS:

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Some experiments require local GPU access, local model serving, vLLM-compatible models, Hugging Face model access, or API credentials depending on the model family being evaluated.

## Data policy

This repository intentionally does not include:

- raw MedQA, MedMCQA, or PubMedQA examples
- raw benchmark question text
- raw model-output JSONL files
- model weights
- Hugging Face cache files
- generated result folders
- paper PDFs, LaTeX build files, figures, or tables

Users should obtain benchmark datasets from their official sources and configure local paths before running the scripts.

## Typical workflow

A typical reproduction workflow is:

    bash scripts/setup/00_install.sh
    bash scripts/setup/01_prepare_data.sh
    bash scripts/setup/02_smoke_test.sh
    bash scripts/original/03_run_main_models.sh
    bash scripts/analysis/03_audit_and_reliability.sh

For larger or expanded evaluations, use the scripts in:

    scripts/scaled/
    scripts/expanded_evaluation/
    scripts/claude/
    scripts/medgemma/

Local paths, model names, GPU IDs, ports, and API credentials may need to be adjusted for each compute environment.

## Configuration

Model and runtime settings are organized under:

    configs/models.yaml

Before running experiments, check that this file matches your local model names, hardware, inference backend, and server ports.

## Reproducibility notes

This is a code-first reproducibility release. It provides scripts and evaluation utilities while excluding large generated artifacts and restricted dataset or model content. This keeps the repository lightweight and avoids redistributing benchmark text or model weights.

