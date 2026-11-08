import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl
from decouple import config

from src.data.annotations import load_annotations, load_split_clips
from src.evaluation.coco_metrics import MAX_TEST_WIDTH, evaluate_detections

logger = logging.getLogger(__name__)

DATA_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "data.toml"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "results.csv"
PREDICTIONS_FILE = "val_predictions.parquet"
METRICS_FILE = "metrics.json"
REPORTED_FRAME_STEP = 1
TRAINING_FIELDS = (
    "config",
    "seed",
    "commit",
    "is_dirty",
    "machine",
    "frame_step",
    "epochs",
    "effective_batch_size",
    "gpu_hours",
    "wandb_url",
    "notes",
)
EVALUATION_FIELDS = (
    "map",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    f"map_max_width_{MAX_TEST_WIDTH}",
    f"ap_small_max_width_{MAX_TEST_WIDTH}",
)
RESULTS_COLUMNS = (
    "experiment",
    "config",
    "run_dir",
    "seed",
    "commit",
    "is_dirty",
    "machine",
    "frame_step",
    "epochs",
    "effective_batch_size",
    "gpu_hours",
    *EVALUATION_FIELDS,
    "ms_per_image",
    "public_lb",
    "wandb_url",
    "notes",
)
METRIC_DECIMALS = 4


def find_run_dir(results_path: Path, experiment: str) -> Path:
    """Look up the run directory of an experiment in `experiments/results.csv`.

    Args:
        results_path: Path of `experiments/results.csv`.
        experiment: Experiment name, e.g. `rtdetr-baseline`.

    Returns:
        The `run_dir` of the experiment's only row, relative to the repository root.
    """
    rows = pl.read_csv(results_path, columns=["experiment", "run_dir"]).filter(
        pl.col("experiment") == experiment
    )
    if rows.height != 1:
        raise ValueError(f"{results_path} has {rows.height} rows for {experiment!r}, expected 1")
    return Path(rows["run_dir"].item())


def check_full_val_inference(metrics_path: Path, val_frames: int) -> None:
    """Check that a run's `inference` record covers all val frames at frame step 1.

    Args:
        metrics_path: Path of the run's `metrics.json`.
        val_frames: Number of val frames.
    """
    inference = json.loads(metrics_path.read_text()).get("inference", {})
    expected = {"split": "val", "frame_step": REPORTED_FRAME_STEP, "num_frames": val_frames}
    recorded = {key: inference.get(key) for key in expected}
    if recorded != expected:
        raise ValueError(
            f"{metrics_path} records inference {recorded}, evaluation needs {expected}"
        )


def evaluate_run(run_dir: Path, data_root: Path, data_config_path: Path) -> dict:
    """Evaluate the validation predictions of a run and store them in its `metrics.json`.

    The `inference` section of `metrics.json` must record `split` `val`, `frame_step` 1,
    and `num_frames` equal to the number of val frames, so that frames without rows in the
    predictions are frames without detections.

    Args:
        run_dir: Run directory with `metrics.json` and `val_predictions.parquet` (real
            detections only: `image_id`, 0-based `class_id`, `score`, pixel `x`, `y`, `w`, `h`).
        data_root: Dataset root with the provided splits.
        data_config_path: Path of `config/data.toml` with `[splits]`.

    Returns:
        The updated `metrics.json` content; the shared evaluator output is under
        `evaluation`, other keys are kept.
    """
    tables = load_annotations(data_root, load_split_clips(data_config_path))
    val_frames = tables.images.filter(pl.col("split") == "val").height
    metrics_path = run_dir / METRICS_FILE
    check_full_val_inference(metrics_path, val_frames)
    metrics = json.loads(metrics_path.read_text())
    predictions = pl.read_parquet(run_dir / PREDICTIONS_FILE)
    metrics["evaluation"] = evaluate_detections(tables, "val", predictions)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def format_results_row(run_dir: Path, metrics: dict) -> str:
    """Format the `experiments/results.csv` header and row of an evaluated run.

    Args:
        run_dir: Run directory `runs/<experiment>/<timestamp>`.
        metrics: Content of the run's `metrics.json` with `training` (all `TRAINING_FIELDS`),
            `inference` (`ms_per_image`), and `evaluation` sections.

    Returns:
        CSV text with the header line and one row; `public_lb` is left empty.
    """
    training = metrics.get("training", {})
    inference = metrics.get("inference", {})
    missing = [f"training.{field}" for field in TRAINING_FIELDS if field not in training]
    if "ms_per_image" not in inference:
        missing.append("inference.ms_per_image")
    if missing:
        raise ValueError(f"{run_dir / METRICS_FILE} lacks {missing} for the results.csv row")
    evaluation = metrics["evaluation"]
    row = training | {
        "experiment": run_dir.parent.name,
        "run_dir": run_dir.as_posix(),
        "ms_per_image": inference["ms_per_image"],
        "public_lb": None,
    }
    row |= {
        field: None if evaluation[field] is None else round(evaluation[field], METRIC_DECIMALS)
        for field in EVALUATION_FIELDS
    }
    return pl.DataFrame([{column: row[column] for column in RESULTS_COLUMNS}]).write_csv()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Evaluate a run on all validation frames.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory to evaluate.")
    args = parser.parse_args()
    metrics = evaluate_run(args.run, Path(config("DATA_ROOT")), DATA_CONFIG_PATH)
    evaluation = metrics["evaluation"]
    logger.info(
        "mAP %s, AP50 %s, AP_small %s on %d frames; wrote %s",
        evaluation["map"],
        evaluation["map50"],
        evaluation["ap_small"],
        evaluation["frames"],
        args.run / METRICS_FILE,
    )
    logger.info("results.csv row:\n%s", format_results_row(args.run, metrics))


if __name__ == "__main__":
    main()
