# HypothesisMed

HypothesisMed is a reproducibility repository for an inference-time biomedical question answering reliability pipeline. The project evaluates whether answer selection, structured output compliance, SPACE-label reporting, parse coverage, and false-commitment behavior should be treated as separate evaluation axes for biomedical multiple-choice question answering.

The repository accompanies the paper:

**HypothesisMed: Inference-Time Answer Fusion and Structured Hypothesis-Space Reporting for Biomedical Question Answering**

## Overview

Biomedical question answering with large language models is usually evaluated by final answer accuracy. HypothesisMed extends this evaluation by adding structured reliability reporting.

The proposed pipeline combines:

1. Direct prompting
2. Chain-of-thought prompting
3. HypothesisMed-v3 structured prompting
4. Majority-based answer fusion
5. SPACE-label reporting

SPACE labels describe the status of the answer space:

- `VALID`: The answer options contain a medically supported best answer.
- `INCOMPLETE`: The answer options appear insufficient or missing the best answer.
- `CONTRADICTED`: The answer options appear internally inconsistent or medically contradictory.

For the original benchmark evaluations in this repository, the gold SPACE label is treated as `VALID`, because the benchmark items are standard multiple-choice QA examples.

## Main Claim

Across MedQA, MedMCQA, and PubMedQA, with 1,000 examples per dataset, the proposed answer-fusion plus HypothesisMed-v3 SPACE pipeline improves weighted answer accuracy over each model's best direct or chain-of-thought baseline while adding structured reliability observability.

The main finding is not a universal state-of-the-art claim. Instead, the repository supports the claim that answer accuracy, parseability, structured reliability reporting, and false-commitment behavior are separable model capabilities.

## Evaluated Models

The experiments include four open-weight models:

- Qwen2.5-7B-Instruct
- Phi-4-mini-instruct
- DeepSeek-R1-Distill-Qwen-32B
- BioMistral-7B

Model snapshot identifiers used in the final runs are listed in:

```text
reproducibility/model_snapshots.txt
```

## Evaluated Datasets

The evaluation uses three biomedical question answering datasets:

- MedQA
- MedMCQA
- PubMedQA

Each dataset is evaluated using 1,000 examples.

Raw benchmark datasets are not included in this repository. Users should obtain the datasets from their official sources and run the preprocessing and evaluation scripts locally.

## Repository Structure

```text
.
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── methods/
│   ├── models/
│   └── utils/
├── experiments/
├── scripts/
├── results/
│   ├── final_4model_paper/
│   ├── final_png_only_assets/
│   │   ├── figures/
│   │   └── tables/
│   └── final_claim_validation/
├── paper_assests/
│   ├── figures/
│   └── tables/
└── reproducibility/
```

## Important Files

### Final paper tables

- `results/final_4model_paper/main_4model_aggregate_table.csv`
- `results/final_4model_paper/deltas_vs_best_direct_cot_baseline.csv`
- `results/final_4model_paper/proposed_per_dataset_4model_table.csv`

### Final paper figures

- `results/final_png_only_assets/figures/`

### Final paper table assets

- `results/final_png_only_assets/tables/`

### Reproducibility metadata

- `reproducibility/inference_and_compute_settings.txt`
- `reproducibility/model_snapshots.txt`
- `reproducibility/python_environment_freeze.txt`
- `reproducibility/data_policy.txt`
- `reproducibility/artifact_manifest.txt`

## Inference and Evaluation Settings

The final reported experiments used the following settings:

- Inference engine: vLLM
- Precision: bfloat16
- Maximum context length: 4096
- Maximum generation length: 512 tokens
- Temperature: 0.0
- Top-p: 1.0
- Seed: 42
- Fusion rule: majority vote with deterministic parseable fallback
- Fallback order: chain-of-thought, direct prompting, HypothesisMed-v3
- False commitment threshold: confidence >= 0.5

Full compute and software settings are provided in:

- `reproducibility/inference_and_compute_settings.txt`

## Metrics

The repository reports the following metrics:

- Answer accuracy: fraction of examples where the predicted answer matches the gold answer.
- Parse coverage: fraction of examples where a valid answer can be extracted from model output.
- SPACE coverage: fraction of examples where a valid SPACE label can be extracted.
- SPACE accuracy: agreement between extracted SPACE label and gold SPACE label when defined.
- False commitment: high-confidence wrong commitment behavior, using confidence >= 0.5 as the high-confidence threshold.

Lower false commitment is better. Higher answer accuracy, parse coverage, SPACE coverage, and SPACE accuracy are better.

## Final Aggregate Results

The main aggregate result table is available here:

- `results/final_4model_paper/main_4model_aggregate_table.csv`

The delta table comparing the proposed method against each model's best direct or chain-of-thought baseline is available here:

- `results/final_4model_paper/deltas_vs_best_direct_cot_baseline.csv`

The per-dataset proposed-method result table is available here:

- `results/final_4model_paper/proposed_per_dataset_4model_table.csv`

## Paper Assets

The `paper_assests/` directory contains copies of the final paper figures and tables for direct use in LaTeX or Overleaf:

- `paper_assests/figures/`
- `paper_assests/tables/`

The spelling `paper_assests` is kept intentionally to match the paper source path used during manuscript preparation.

## Data Policy

This repository does not include:

- Raw MedQA examples
- Raw MedMCQA examples
- Raw PubMedQA examples
- Raw model-output JSONL files
- Model weights
- Hugging Face cache files

This is intentional. Raw datasets should be obtained from their official sources. Raw model outputs may contain benchmark question text and are therefore excluded from the public repository.

See:

- `reproducibility/data_policy.txt`

## Reproducing the Evaluation

A typical reproduction workflow is:

1. Install dependencies in a Python environment.
2. Obtain MedQA, MedMCQA, and PubMedQA from their official sources.
3. Prepare the transformed dataset files using the scripts in `scripts/` and `src/data/`.
4. Run the experiment scripts in `experiments/`.
5. Reparse and summarize model outputs using the scripts in `scripts/`.
6. Regenerate final tables and figures from the summarized result files.

The exact dataset acquisition paths and local paths may need to be adjusted for the user's environment.


## License

## License

License information has not yet been finalized. Until a license is added, this repository is provided for transparency and reproducibility review only. Raw datasets, benchmark question text, model weights, and third-party resources are not included and remain governed by their respective licenses or terms of use.
