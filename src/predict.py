import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

import polars as pl
from decouple import config

from src.data.annotations import load_annotations, load_split_clips, load_test_images
from src.evaluate import (
    DATA_CONFIG_PATH,
    METRICS_FILE,
    PREDICTIONS_FILE,
    REPORTED_FRAME_STEP,
)
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.frames import predict_frames
from src.models.registry import build_model
from src.submission.validate import validate_submission
from src.submission.writer import write_submission
from src.training.experiment import CONFIG_FILE

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "best.pt"
TEST_PREDICTIONS_FILE = "predictions.csv"
PREDICTION_SPLITS = ("val", "test")


def write_val_predictions(
    run_dir: Path, predictions: pl.DataFrame, val_images: pl.DataFrame, ms_per_image: float
) -> None:
    """Write the predictions of all val frames and record the inference in `metrics.json`.

    Args:
        run_dir: Run directory with the `metrics.json` written by training.
        predictions: Predictions on all val frames with `PREDICTION_COLUMNS`.
        val_images: Image table of all val frames with `image_id`.
        ms_per_image: Mean inference time in ms per image.
    """
    validate_predictions(predictions, val_images)
    metrics_path = run_dir / METRICS_FILE
    metrics = json.loads(metrics_path.read_text())
    predictions.select(PREDICTION_COLUMNS).write_parquet(run_dir / PREDICTIONS_FILE)
    metrics["inference"] = {
        "split": "val",
        "frame_step": REPORTED_FRAME_STEP,
        "num_frames": val_images.height,
        "ms_per_image": ms_per_image,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def write_test_predictions(
    run_dir: Path, predictions: pl.DataFrame, test_images: pl.DataFrame
) -> Path:
    """Write test predictions as a Kaggle submission in the run directory and validate it.

    Args:
        run_dir: Run directory.
        predictions: Predictions on the test frames, as accepted by `format_submission`.
        test_images: Table of all test frames from `load_test_images`.

    Returns:
        Path of the written `predictions.csv`.
    """
    csv_path = run_dir / TEST_PREDICTIONS_FILE
    write_submission(predictions, test_images, csv_path)
    validate_submission(csv_path, test_images)
    return csv_path


def run_prediction(run_dir: Path, split: str, data_root: Path, data_config_path: Path) -> Path:
    """Predict all frames of a split with the run's `best.pt` and write the outputs.

    Args:
        run_dir: Run directory with the resolved `config.toml` and `best.pt`.
        split: `val` for `val_predictions.parquet`, `test` for `predictions.csv`.
        data_root: Dataset root with the provided splits and `test_images.csv`.
        data_config_path: Path of `config/data.toml` with `[splits]`.

    Returns:
        Path of the written predictions file.
    """
    if split not in PREDICTION_SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {PREDICTION_SPLITS}")
    config = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    model = build_model(config)
    if split == "test":
        test_images = load_test_images(data_root)
        predictions, _ = predict_frames(model, run_dir / CHECKPOINT_FILE, test_images, config)
        return write_test_predictions(run_dir, predictions, test_images)
    tables = load_annotations(data_root, load_split_clips(data_config_path))
    val_images = tables.images.filter(pl.col("split") == "val")
    predictions, ms_per_image = predict_frames(model, run_dir / CHECKPOINT_FILE, val_images, config)
    write_val_predictions(run_dir, predictions, val_images, ms_per_image)
    return run_dir / PREDICTIONS_FILE


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Predict val or test frames with a run.")
    parser.add_argument("--run", type=Path, required=True, help="runs/<experiment>/<timestamp>")
    parser.add_argument("--split", choices=PREDICTION_SPLITS, required=True)
    args = parser.parse_args()
    output = run_prediction(args.run, args.split, Path(config("DATA_ROOT")), DATA_CONFIG_PATH)
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
