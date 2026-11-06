#!/usr/bin/env bash
# Usage: scripts/run_experiment.sh config/<family>/<name>.toml
# Trains the experiment, then runs scripts/finish_run.sh on its new run directory.
# Submit separately after reviewing the val metrics: scripts/finish_run.sh <run> --submit
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 config/<family>/<name>.toml" >&2
    exit 2
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=$(realpath --relative-to="$ROOT" "$1")
cd "$ROOT"

TRAIN_LOG=$(mktemp)
trap 'rm -f "$TRAIN_LOG"' EXIT
PYTHONUNBUFFERED=1 uv run python -m src.train --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

RUN_ABS=$(sed -n 's/^.*\] __main__: Training .*, run directory //p' "$TRAIN_LOG")
if [[ -z "$RUN_ABS" || ! -d "$RUN_ABS" ]]; then
    echo "Could not find the run directory in the training log" >&2
    exit 1
fi
RUN=$(realpath --relative-to="$ROOT" "$RUN_ABS")
scripts/finish_run.sh "$RUN"
