#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.data.load_datasets
python -m src.data.transform
