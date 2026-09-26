# Reproducibility Manifest

## Public-release scope

This repository publishes source code, prompts, parser, reproducibility configurations, frozen generation settings, protocol metadata, deterministic analysis code, and documentation. It does not publish experimental results or benchmark content.

## Hashes

See `configs/frozen_hashes.json` for prompt, parser, membership, generated-input, and protocol-file hashes.

## Public portability edits

The public copies of generation and analysis scripts were edited only to remove private absolute local paths and private Hugging Face cache paths. Model loading in public generation scripts uses model IDs plus exact frozen revision SHAs rather than local snapshot directories.

The public copy of the SC-3 analysis script also omits final numerical audit-anchor constants because those constants are result values. The SC-3 derivation, paired bootstrap, McNemar diagnostics, Holm logic, budget accounting, and reporting estimators are otherwise preserved.

## Excluded materials

The repository excludes benchmark item text, gold labels paired with items, generated model outputs, JSONL generation files, result CSV/Markdown/Parquet files, run logs, model weights, tokenizer downloads, caches, and environment directories.
