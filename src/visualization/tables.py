import json
from pathlib import Path

import polars as pl

from src.evaluate import METRICS_FILE


def load_run_histories(run_dirs: dict[str, Path]) -> pl.DataFrame:
    """Read the per-epoch training history of several runs into one long table.

    Args:
        run_dirs: Mapping from experiment name to its run directory with `metrics.json`.

    Returns:
        One row per run and epoch with `experiment` first and the history fields of
        `metrics.json` (`epoch`, losses, monitoring metrics, and timings) after it.
    """
    histories = []
    for experiment, run_dir in run_dirs.items():
        metrics = json.loads((run_dir / METRICS_FILE).read_text())
        histories.append(
            pl.DataFrame(metrics["training"]["history"]).select(
                pl.lit(experiment).alias("experiment"), pl.all()
            )
        )
    return pl.concat(histories, how="diagonal_relaxed")


def compare_with_reference(
    results: pl.DataFrame, reference: str, metrics: list[str]
) -> pl.DataFrame:
    """Add the difference of each metric from a reference experiment to a results table.

    Args:
        results: Rows of `experiments/results.csv` with `experiment` and the metric columns.
        reference: Experiment whose row is subtracted, e.g. the family baseline.
        metrics: Metric columns to compare.

    Returns:
        `experiment`, the metric columns, and one `delta_<metric>` column per metric, in the
        row order of `results`.
    """
    reference_row = results.filter(pl.col("experiment") == reference)
    if reference_row.height != 1:
        raise ValueError(f"Expected one results row for {reference!r}, got {reference_row.height}")
    return results.select(
        "experiment",
        *metrics,
        *((pl.col(m) - reference_row[m].item()).alias(f"delta_{m}") for m in metrics),
    )
