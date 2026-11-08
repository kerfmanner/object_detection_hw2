import argparse
import json
import logging
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import polars as pl
from decouple import config

from src.data.annotations import load_annotations, load_split_clips, load_test_images
from src.evaluate import (
    DATA_CONFIG_PATH,
    METRICS_FILE,
    PREDICTIONS_FILE,
    REPORTED_FRAME_STEP,
    RESULTS_PATH,
    find_run_dir,
)
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.frames import predict_frames
from src.models.registry import build_model
from src.submission.validate import validate_submission
from src.submission.writer import write_submission
from src.train import RUNS_ROOT
from src.training.experiment import (
    CONFIG_FILE,
    REPO_ROOT,
    create_run_dir,
    describe_run,
    format_toml,
    merge_config,
    name_experiment,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "best.pt"
TEST_PREDICTIONS_FILE = "predictions.csv"
PREDICTION_SPLITS = ("val", "test")
SOURCE_KEY = "source"
INFERENCE_CONFIG_KEYS = (SOURCE_KEY, "min_size", "max_size", "imgsz", "inference")


def load_run_config(run_dir: Path) -> dict:
    """Read the resolved configuration of a run for prediction.

    Inference options added to the family baseline after the run was resolved take their
    baseline value, which reproduces the earlier behaviour; options the run records are kept.

    Args:
        run_dir: Run directory with the resolved `config.toml`, whose `run.baseline` names
            the family baseline.

    Returns:
        The resolved configuration with a complete `[inference]` table.
    """
    config = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    baseline = tomllib.loads(Path(config["run"]["baseline"]).read_text())
    return config | {"inference": baseline["inference"] | config["inference"]}


def create_inference_run(config_path: Path, runs_root: Path, results_path: Path) -> Path:
    """Create a run that predicts with the checkpoint of an earlier run and other inference
    settings.

    The inference config `config/<family>/infer_<name>.toml` names its `source` experiment,
    which must have one row in `results.csv`, and overrides only inference settings
    (`INFERENCE_CONFIG_KEYS`). The new run `runs/<family>-infer_<name>/<timestamp>/` gets the
    source configuration merged with the overrides as `config.toml` (with `run.source_run`),
    a relative `best.pt` symlink to the source checkpoint, and a `metrics.json` whose
    `training` section is the source's with this config, its first-line note, the current
    commit and machine, and `source_run`, `source_config`, `source_commit`, and
    `source_notes`. `run_prediction`, `src.evaluate`, and `scripts/finish_run.sh` then work
    on it as on a trained run.

    Args:
        config_path: Path of the inference config.
        runs_root: Root directory of all runs, e.g. `runs/`.
        results_path: Path of `experiments/results.csv`.

    Returns:
        The new run directory.
    """
    overrides = tomllib.loads(config_path.read_text())
    unknown = sorted(set(overrides) - set(INFERENCE_CONFIG_KEYS))
    if unknown or SOURCE_KEY not in overrides:
        raise ValueError(
            f"{config_path} must set {SOURCE_KEY!r} and may only set {INFERENCE_CONFIG_KEYS}, "
            f"found {unknown or 'no source'}"
        )
    source_dir = find_run_dir(results_path, overrides.pop(SOURCE_KEY))
    source_checkpoint = source_dir / CHECKPOINT_FILE
    if not source_checkpoint.is_file():
        raise ValueError(f"{source_checkpoint} does not exist")
    source_config = load_run_config(source_dir)
    if "source_run" in source_config["run"]:
        raise ValueError(f"The source of {config_path} must be a trained run, not {source_dir}")
    family = config_path.parent.name
    if source_config["run"]["family"] != family:
        raise ValueError(
            f"{config_path} is in family {family!r}, its source in "
            f"{source_config['run']['family']!r}"
        )
    config = merge_config(source_config, overrides)
    baseline_seed = tomllib.loads(Path(config["run"]["baseline"]).read_text())["seed"]
    experiment = name_experiment(family, config_path.stem, config["seed"], baseline_seed)
    config["run"] = source_config["run"] | {
        "experiment": experiment,
        "config": config_path.as_posix(),
        "source_run": source_dir.as_posix(),
    }
    run_dir = create_run_dir(runs_root, experiment, datetime.now())
    (run_dir / CONFIG_FILE).write_text(format_toml(config))
    (run_dir / CHECKPOINT_FILE).symlink_to(
        source_checkpoint.resolve().relative_to(run_dir.resolve(), walk_up=True)
    )
    source_training = json.loads((source_dir / METRICS_FILE).read_text())["training"]
    training = (
        source_training
        | describe_run(config, REPO_ROOT)
        | {
            "source_run": source_dir.as_posix(),
            "source_config": source_training["config"],
            "source_commit": source_training["commit"],
            "source_notes": source_training["notes"],
        }
    )
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training}, indent=2) + "\n")
    return run_dir


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
    config = load_run_config(run_dir)
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
    parser = argparse.ArgumentParser(
        description="Predict val or test frames with a run, or create an inference run."
    )
    parser.add_argument("--run", type=Path, help="runs/<experiment>/<timestamp>")
    parser.add_argument("--split", choices=PREDICTION_SPLITS)
    parser.add_argument(
        "--inference-config",
        type=Path,
        help="config/<family>/infer_<name>.toml: create its inference run and predict nothing",
    )
    args = parser.parse_args()
    if args.inference_config is not None:
        if args.run is not None or args.split is not None:
            parser.error("--inference-config takes no --run or --split")
        run_dir = create_inference_run(args.inference_config, RUNS_ROOT, RESULTS_PATH)
        logger.info("Inference run directory %s", run_dir)
        return
    if args.run is None or args.split is None:
        parser.error("--run and --split are required without --inference-config")
    output = run_prediction(args.run, args.split, Path(config("DATA_ROOT")), DATA_CONFIG_PATH)
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
