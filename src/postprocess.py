import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from decouple import config

from src.data.annotations import (
    AnnotationTables,
    load_annotations,
    load_split_clips,
    load_test_images,
)
from src.evaluate import (
    DATA_CONFIG_PATH,
    METRICS_FILE,
    PREDICTIONS_FILE,
    RESULTS_PATH,
    check_full_val_inference,
    find_run_dir,
)
from src.evaluation.coco_metrics import (
    MAX_TEST_WIDTH,
    evaluate_detections,
    evaluate_frame_groups,
)
from src.inference.fusion import fuse_predictions
from src.inference.temporal import rescore_along_tracks
from src.predict import (
    RUNS_ROOT,
    TEST_PREDICTIONS_FILE,
    write_test_predictions,
    write_val_predictions,
)
from src.submission.validate import read_submission_predictions
from src.training.experiment import (
    CONFIG_FILE,
    REPO_ROOT,
    create_run_dir,
    describe_run,
    format_toml,
    load_experiment_config,
    merge_config,
)

logger = logging.getLogger(__name__)

SHARED_TRAINING_FIELDS = ("seed", "frame_step", "epochs", "effective_batch_size")


def postprocess_predictions(
    members: list[pl.DataFrame], images: pl.DataFrame, config: dict
) -> pl.DataFrame:
    """Fuse the predictions of the members, then optionally rescore them along tracks.

    Args:
        members: Prediction tables with `PREDICTION_COLUMNS` of the members on `images`.
        images: Image table with `image_id`, `clip`, and `frame_id` of the predicted frames.
        config: Resolved post-processing configuration with `weights`, `fusion`,
            `iou_thresh`, and the `[temporal]` table of `config/postprocess/baseline.toml`.

    Returns:
        Predictions with `PREDICTION_COLUMNS`, at most 500 per frame.
    """
    if config["fusion"] == "none":
        if len(members) != 1:
            raise ValueError(f"fusion none needs exactly one member, got {len(members)}")
        predictions = members[0]
    else:
        predictions = fuse_predictions(
            members, config["weights"], config["fusion"], config["iou_thresh"]
        )
    temporal = config["temporal"]
    if not temporal["enabled"]:
        return predictions
    return rescore_along_tracks(
        predictions,
        images,
        temporal["link_iou"],
        temporal["min_score"],
        temporal["weight"],
        temporal["statistic"],
    )


def evaluate_postprocess_settings(
    tables: AnnotationTables,
    members: list[pl.DataFrame],
    base_config: dict,
    settings: list[dict],
    clip_folds: dict[str, str],
) -> pl.DataFrame:
    """Evaluate post-processing settings on the val predictions of the members.

    Each setting is merged over `base_config` (`merge_config`), applied with
    `postprocess_predictions`, and scored with the shared evaluator on all val frames and
    separately on each fold of val clips, so that a setting chosen on all of val can be
    checked for consistency across disjoint clips.

    Args:
        tables: Annotation tables with all val frames.
        members: Val predictions of the members, in the order of the config's `weights`.
        base_config: Resolved post-processing configuration the settings change.
        settings: Overrides of `base_config`, e.g. `{"fusion": "wbf", "iou_thresh": 0.6}`.
        clip_folds: Fold name of every val clip.

    Returns:
        One row per setting with `setting` (the override as JSON), `map`, `map50`,
        `ap_small`, `map_max_width_1920`, one `ap_<class>` column per class, and
        `map_<fold>` per fold.
    """
    val_images = tables.images.filter(pl.col("split") == "val")
    frame_groups = val_images.select(
        "image_id", pl.col("clip").replace_strict(clip_folds).alias("group")
    )
    rows = []
    for setting in settings:
        config = merge_config(base_config, setting)
        predictions = postprocess_predictions(members, val_images, config)
        metrics = evaluate_detections(tables, "val", predictions)
        folds = evaluate_frame_groups(tables, "val", predictions, frame_groups)
        rows.append(
            {"setting": json.dumps(setting)}
            | {key: metrics[key] for key in ("map", "map50", "ap_small")}
            | {f"map_max_width_{MAX_TEST_WIDTH}": metrics[f"map_max_width_{MAX_TEST_WIDTH}"]}
            | {f"ap_{name}": ap for name, ap in metrics["per_class_ap"].items()}
            | {f"map_{group}": value for group, value in folds.select("group", "map").rows()}
        )
    return pl.DataFrame(rows)


def describe_members(config: dict, member_dirs: list[Path], member_metrics: list[dict]) -> dict:
    """Describe a post-processing run and the training of its members for `metrics.json`.

    Fields the members share (seed, frame step, epochs, effective batch size) keep their
    value and are None otherwise. `gpu_hours` sums the training time of the distinct trained
    models: an inference run (`source_run`) reuses the model of its source run.

    Args:
        config: Resolved post-processing configuration.
        member_dirs: Run directory of each member.
        member_metrics: `metrics.json` content of each member.

    Returns:
        The `training` section with the fields of `describe_run`, the shared fields,
        `gpu_hours`, `wandb_url` None (no W&B run), and `members` (experiment, run_dir,
        weight of each member).
    """
    trainings = [metrics["training"] for metrics in member_metrics]
    shared = {}
    for field in SHARED_TRAINING_FIELDS:
        values = {training[field] for training in trainings}
        shared[field] = values.pop() if len(values) == 1 else None
    model_hours = {
        training["source_run"] if "source_run" in training else run_dir.as_posix(): training[
            "gpu_hours"
        ]
        for run_dir, training in zip(member_dirs, trainings, strict=True)
    }
    return (
        describe_run(config, REPO_ROOT)
        | shared
        | {
            "gpu_hours": sum(model_hours.values()),
            "wandb_url": None,
            "members": [
                {"experiment": experiment, "run_dir": run_dir.as_posix(), "weight": weight}
                for experiment, run_dir, weight in zip(
                    config["members"], member_dirs, config["weights"], strict=True
                )
            ],
        }
    )


def run_postprocess(
    config_path: Path, data_root: Path, data_config_path: Path, runs_root: Path, results_path: Path
) -> Path:
    """Post-process the saved val and test predictions of member runs into a new run.

    The members are looked up by experiment name in `results.csv`. Their val predictions
    (`val_predictions.parquet`, all val frames) and test predictions (`predictions.csv`) go
    through `postprocess_predictions`. The new run directory gets the resolved `config.toml`,
    `metrics.json` with `training` (`describe_members`) and `inference`, the val predictions,
    and a validated `predictions.csv`, so that `src.evaluate` works on it unchanged. Its
    ms/image is the sum of the members' ms/image plus the post-processing time per val frame.

    Args:
        config_path: Path `config/postprocess/<name>.toml`.
        data_root: Dataset root with the provided splits and `test_images.csv`.
        data_config_path: Path of `config/data.toml` with `[splits]`.
        runs_root: Root directory of all runs, e.g. `runs/`.
        results_path: Path of `experiments/results.csv`.

    Returns:
        The new run directory.
    """
    config = load_experiment_config(config_path)
    if len(config["weights"]) != len(config["members"]):
        raise ValueError(f"{config_path} needs one weight per member")
    member_dirs = [find_run_dir(results_path, experiment) for experiment in config["members"]]
    tables = load_annotations(data_root, load_split_clips(data_config_path))
    val_images = tables.images.filter(pl.col("split") == "val")
    for member_dir in member_dirs:
        check_full_val_inference(member_dir / METRICS_FILE, val_images.height)
    member_metrics = [json.loads((d / METRICS_FILE).read_text()) for d in member_dirs]
    val_members = [pl.read_parquet(d / PREDICTIONS_FILE) for d in member_dirs]
    test_images = load_test_images(data_root)
    test_members = [
        read_submission_predictions(d / TEST_PREDICTIONS_FILE, test_images) for d in member_dirs
    ]
    run_dir = create_run_dir(runs_root, config["run"]["experiment"], datetime.now())
    (run_dir / CONFIG_FILE).write_text(format_toml(config))
    training = describe_members(config, member_dirs, member_metrics)
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training}, indent=2) + "\n")
    started = time.perf_counter()
    val_predictions = postprocess_predictions(val_members, val_images, config)
    seconds = time.perf_counter() - started
    ms_per_image = sum(m["inference"]["ms_per_image"] for m in member_metrics) + (
        1000 * seconds / val_images.height
    )
    write_val_predictions(run_dir, val_predictions, val_images, ms_per_image)
    test_predictions = postprocess_predictions(test_members, test_images, config)
    write_test_predictions(run_dir, test_predictions, test_images)
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(
        description="Fuse or rescore the saved predictions of runs into a new run."
    )
    parser.add_argument("--config", type=Path, required=True, help="config/postprocess/<name>.toml")
    args = parser.parse_args()
    run_dir = run_postprocess(
        args.config, Path(config("DATA_ROOT")), DATA_CONFIG_PATH, RUNS_ROOT, RESULTS_PATH
    )
    logger.info("Post-processing run directory %s", run_dir)


if __name__ == "__main__":
    main()
