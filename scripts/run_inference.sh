#!/usr/bin/env bash
# Usage: scripts/run_inference.sh config/<family>/infer_<name>.toml
# Creates the inference run of the config (the best.pt of its source run with other inference
# settings), then runs scripts/finish_run.sh on it: val prediction, evaluation, test
# prediction, and the results.csv row. Submit separately: scripts/finish_run.sh <run> --submit
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 config/<family>/infer_<name>.toml" >&2
    exit 2
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=$(realpath --relative-to="$ROOT" "$1")
cd "$ROOT"

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
PYTHONUNBUFFERED=1 uv run python -m src.predict --inference-config "$CONFIG" 2>&1 | tee "$LOG"

RUN_ABS=$(sed -n 's/^.*\] __main__: Inference run directory //p' "$LOG")
if [[ -z "$RUN_ABS" || ! -d "$RUN_ABS" ]]; then
    echo "Could not find the run directory in the log" >&2
    exit 1
fi
RUN=$(realpath --no-symlinks --relative-to="$ROOT" "$RUN_ABS")
scripts/finish_run.sh "$RUN"
