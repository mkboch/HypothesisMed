# Not All Inference-Time Aggregation Is Equal:
# Heterogeneous Prompt Fusion versus Self-Consistency in Biomedical Question Answering

## Overview

This repository contains the public reproducibility materials for the HypothesisMed paper. It is a code, prompt, parser, configuration, and protocol repository. It intentionally does not redistribute benchmark question text, gold labels paired with benchmark items, generated model outputs, or result tables.

Repository URL: https://github.com/mkboch/HypothesisMed

## Study design

The study evaluates inference-time aggregation strategies for biomedical multiple-choice question answering under a prospectively frozen five-model benchmark. The primary benchmark compares Direct, Chain-of-Thought, Structured/HypothesisMed, and a deterministic heterogeneous Fusion rule. A prospective SC-3 control evaluates same-prompt three-sample Chain-of-Thought self-consistency with matched model-call count.

## Repository contents

- `prompts/`: exact frozen prompt templates.
- `src/evaluation/parser.py`: exact frozen parser.
- `configs/`: public model panel, generation settings, serving conditions, and hashes.
- `scripts/primary/`: primary native-chat generation runner.
- `scripts/sc3/`: SC-3 native-chat generation runner.
- `scripts/analysis/`: deterministic analysis code for primary, mechanistic, and SC-3 analyses.
- `protocol/`: protocol summaries and reproducibility manifest.
- `docs/`: data availability and reproducibility notes.

## Frozen model panel

See `configs/model_panel.json` for the exact Hugging Face model IDs and full revision SHAs. The panel consists of Qwen3.6-35B-A3B, Gemma 4 31B IT, MedGemma 27B IT, Qwen2.5 7B Instruct, and Phi-4-mini-instruct.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Large-model generation requires appropriate GPU resources and access to the referenced model repositories under their licenses and terms.

## Primary benchmark

The primary benchmark uses native-chat serialization, frozen prompts, the frozen parser, and the per-model decoding settings in `configs/primary_generation_settings.json`. Fusion is derived after generation by strict majority vote over Direct, CoT, and Structured predictions, with Direct > CoT > Structured fallback when all three disagree.

## SC-3 prospective control

The SC-3 control uses the same frozen CoT prompt with three prospective stochastic samples per item. The frozen seeds, decoding settings, and deterministic majority/fallback rule are in `configs/sc3_generation_settings.json` and `protocol/SC3_PROTOCOL.md`.

## Reproducibility hashes

Prompt and parser hashes are recorded in `configs/frozen_hashes.json` and `protocol/REPRODUCIBILITY_MANIFEST.md`.

## Data availability

Benchmark datasets and generated outputs are not redistributed here. See `docs/DATA_AVAILABILITY.md`.

## Citation

Please cite the paper and repository metadata in `CITATION.cff`.

## License

See `LICENSE`.
