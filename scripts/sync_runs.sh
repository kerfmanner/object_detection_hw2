#!/usr/bin/env bash
# Copy the small files of every run from the PC into the laptop's runs/ (no checkpoints).
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='config.toml' \
    --include='metrics.json' \
    --include='val_predictions.parquet' \
    --include='evaluate.log' \
    --include='results.csv' \
    --exclude='*' \
    pc:~/projects/object_detection_hw2/runs/ runs/
