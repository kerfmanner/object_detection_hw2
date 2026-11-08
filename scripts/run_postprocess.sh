#!/usr/bin/env bash
# Usage: scripts/run_postprocess.sh config/postprocess/<name>.toml
# Fuses or rescores the saved val and test predictions of the config's member runs into a new
# run, evaluates it, validates its CSV, then appends its results.csv row.
# Submit separately: scripts/finish_run.sh <run> --submit
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 config/postprocess/<name>.toml" >&2
    exit 2
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=$(realpath --relative-to="$ROOT" "$1")
cd "$ROOT"
RESULTS=experiments/results.csv

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
PYTHONUNBUFFERED=1 uv run python -m src.postprocess --config "$CONFIG" 2>&1 | tee "$LOG"
RUN_ABS=$(sed -n 's/^.*\] __main__: Post-processing run directory //p' "$LOG")
if [[ -z "$RUN_ABS" || ! -d "$RUN_ABS" ]]; then
    echo "Could not find the run directory in the log" >&2
    exit 1
fi
RUN=$(realpath --no-symlinks --relative-to="$ROOT" "$RUN_ABS")

uv run python -m src.evaluate --run "$RUN" 2>&1 | tee "$RUN/evaluate.log"
HEADER=$(sed -n '/^experiment,/p' "$RUN/evaluate.log")
ROW=$(sed -n '/^experiment,/{n;p;}' "$RUN/evaluate.log")
if [[ -z "$ROW" || "$HEADER" != "$(head -n 1 "$RESULTS")" ]]; then
    echo "The results.csv row in $RUN/evaluate.log is missing or has other columns" >&2
    exit 1
fi
uv run python -m src.submission.validate "$RUN/predictions.csv"
echo "$ROW" >> "$RESULTS"
echo "Appended the $RUN row to $RESULTS"
echo "Done: $(basename "$(dirname "$RUN")")"
