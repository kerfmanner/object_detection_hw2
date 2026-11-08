#!/usr/bin/env bash
# Usage: scripts/run_queue.sh <kind>:<config> [<kind>:<config> ...]
# Runs the items in order: train (scripts/run_experiment.sh), infer (scripts/run_inference.sh),
# or post (scripts/run_postprocess.sh). After each item that appends a results.csv row, commits
# only experiments/results.csv with the measured mAP. A failed item is reported and skipped;
# the exit status is non-zero if any item failed.
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 train|infer|post:<config> ..." >&2
    exit 2
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
RESULTS=experiments/results.csv
FAILED=()

for ITEM in "$@"; do
    KIND=${ITEM%%:*}
    CONFIG=${ITEM#*:}
    case "$KIND" in
        train) SCRIPT=scripts/run_experiment.sh ;;
        infer) SCRIPT=scripts/run_inference.sh ;;
        post) SCRIPT=scripts/run_postprocess.sh ;;
        *) echo "Unknown kind $KIND in $ITEM" >&2; FAILED+=("$ITEM"); continue ;;
    esac
    ROWS_BEFORE=$(wc -l < "$RESULTS")
    echo "=== $(date '+%Y-%m-%dT%H:%M:%S') $ITEM"
    if ! "$SCRIPT" "$CONFIG"; then
        echo "=== FAILED: $ITEM" >&2
        FAILED+=("$ITEM")
        continue
    fi
    if [[ $(wc -l < "$RESULTS") -le $ROWS_BEFORE ]]; then
        echo "=== FAILED: $ITEM appended no results.csv row" >&2
        FAILED+=("$ITEM")
        continue
    fi
    MESSAGE=$(tail -n 1 "$RESULTS" | uv run python -c "
import csv, sys
header = open('$RESULTS').readline().strip().split(',')
row = dict(zip(header, next(csv.reader(sys.stdin))))
print(f\"exp: {row['experiment']} val mAP {row['map']}, AP50 {row['map50']}, AP_small \"
      f\"{row['ap_small']}, mAP <= 1920 px {row['map_max_width_1920']}\")
print()
print(f\"Run {row['run_dir']}, config {row['config']}; row in $RESULTS.\")
")
    git commit -q -m "$MESSAGE" -- "$RESULTS" && git log -1 --format=%s
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed items: ${FAILED[*]}" >&2
    exit 1
fi
echo "All ${#@} items done"
