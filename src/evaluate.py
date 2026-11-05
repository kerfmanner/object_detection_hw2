import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl
from decouple import config

from src.data.annotations import load_annotations, load_split_clips
from src.data.dataset import load_frame_steps, select_frames
from src.evaluation.coco_metrics import (
    MAX_TEST_WIDTH,
    PREDICTION_COLUMNS,
    evaluate_detections,
)

logger = logging.getLogger(__name__)

DATA_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "data.toml"
PREDICTIONS_FILE = "val_predictions.parquet"
METRICS_FILE = "metrics.json"
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


def evaluate_run(run_dir: Path, data_root: Path, data_config_path: Path) -> dict:
    """Evaluate the validation predictions of a run and store them in its `metrics.json`.

    Args:
        run_dir: Run directory with `val_predictions.parquet` (`image_id`, 0-based
            `class_id`, `score`, pixel `x`, `y`, `w`, `h`) with rows for every reported val
            frame; a frame without detections has one row with only `image_id` set.
        data_root: Dataset root with the provided splits.
        data_config_path: Path of `config/data.toml` with `[splits]` and `[frame_steps]`.

    Returns:
        The updated `metrics.json` content; the shared evaluator output is under
        `evaluation`, other keys are kept.
    """
    rows = pl.read_parquet(run_dir / PREDICTIONS_FILE)
    tables = load_annotations(data_root, load_split_clips(data_config_path))
    frame_step = load_frame_steps(data_config_path)["val_reporting"]
    tables = tables._replace(images=select_frames(tables.images, frame_step))
    val_ids = tables.images.filter(pl.col("split") == "val")["image_id"]
    missing = val_ids.filter(~val_ids.is_in(rows["image_id"].implode()))
    if missing.len():
        raise ValueError(
            f"{run_dir / PREDICTIONS_FILE} has no rows for {missing.len()} val frames, "
            f"e.g. image_id {missing[0]}"
        )
    detection_columns = [column for column in PREDICTION_COLUMNS if column != "image_id"]
    predictions = rows.filter(~pl.all_horizontal(pl.col(detection_columns).is_null()))
    evaluation = evaluate_detections(tables, "val", predictions)
    metrics_path = run_dir / METRICS_FILE
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["evaluation"] = evaluation | {"frame_step": frame_step}
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
