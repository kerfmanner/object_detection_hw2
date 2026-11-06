#!/usr/bin/env bash
# Usage: scripts/finish_run.sh runs/<experiment>/<timestamp> [--submit]
# Predicts val, evaluates, predicts test, validates the CSV, then appends the results.csv row.
# With --submit, also copies the CSV to submissions/ and uploads it to Kaggle; on a run that
# already has its row, --submit only validates and uploads.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ($# -eq 2 && "$2" != "--submit") ]]; then
    echo "Usage: $0 runs/<experiment>/<timestamp> [--submit]" >&2
    exit 2
fi
if [[ ! -d "$1" ]]; then
    echo "Run directory $1 does not exist" >&2
    exit 1
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=$(realpath --relative-to="$ROOT" "$1")
cd "$ROOT"

EXP=$(basename "$(dirname "$RUN")")
COMPETITION=multiple-object-video-detection-hw-2-ucu-ai-2026
RESULTS=experiments/results.csv
SUBMISSION=submissions/$EXP.csv
IS_SUBMIT=$([[ $# -eq 2 ]] && echo true || echo false)

HAS_ROW=$(grep -qF ",$RUN," "$RESULTS" && echo true || echo false)
if [[ "$HAS_ROW" == true && "$IS_SUBMIT" == false ]]; then
    echo "$RESULTS already has a row for $RUN; pass --submit to upload it" >&2
    exit 1
fi
if [[ "$IS_SUBMIT" == true && -e "$SUBMISSION" ]]; then
    echo "$SUBMISSION already exists; not overwriting it" >&2
    exit 1
fi

if [[ "$HAS_ROW" == false ]]; then
    uv run python -m src.predict --run "$RUN" --split val
    uv run python -m src.evaluate --run "$RUN" 2>&1 | tee "$RUN/evaluate.log"
    HEADER=$(sed -n '/^experiment,/p' "$RUN/evaluate.log")
    ROW=$(sed -n '/^experiment,/{n;p;}' "$RUN/evaluate.log")
    if [[ -z "$ROW" || "$HEADER" != "$(head -n 1 "$RESULTS")" ]]; then
        echo "The results.csv row in $RUN/evaluate.log is missing or has other columns" >&2
        exit 1
    fi
    uv run python -m src.predict --run "$RUN" --split test
fi
uv run python -m src.submission.validate "$RUN/predictions.csv"
if [[ "$HAS_ROW" == false ]]; then
    echo "$ROW" >> "$RESULTS"
    echo "Appended the $RUN row to $RESULTS"
fi

if [[ "$IS_SUBMIT" == true ]]; then
    mkdir -p submissions
    cp "$RUN/predictions.csv" "$SUBMISSION"
    kaggle competitions submit -c "$COMPETITION" -f "$SUBMISSION" -m "$EXP"
    echo "Submitted $SUBMISSION; fill in public_lb for $RUN in $RESULTS by hand"
fi
echo "Done: $EXP"
