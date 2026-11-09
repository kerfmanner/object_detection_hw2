import io
import json
from pathlib import Path

import polars as pl
import pytest

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.evaluate import (
    METRICS_FILE,
    PREDICTIONS_FILE,
    RESULTS_COLUMNS,
    evaluate_run,
    format_results_row,
)
from src.training.experiment import CONFIG_FILE, format_toml, load_experiment_config
from tests.conftest import SYNTHETIC_SPLIT_CLIPS
from tests.test_smoke import FASTER_RCNN_BASELINE

TRAINING = {
    "config": "config/faster_rcnn/baseline.toml",
    "seed": 42,
    "commit": "abc1234",
    "is_dirty": False,
    "machine": "pc",
    "frame_step": 5,
    "epochs": 12,
    "effective_batch_size": 8,
    "gpu_hours": 1.5,
    "wandb_url": "https://wandb.ai/entity/project/runs/x",
    "notes": "baseline",
    "best_epoch": 9,
}
INFERENCE = {"split": "val", "frame_step": 1, "num_frames": 3, "ms_per_image": 12.3456}


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / METRICS_FILE).write_text(json.dumps(metrics))


@pytest.fixture
def synthetic_data_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "data.toml"
    splits = "\n".join(
        f"{split} = {json.dumps(clips)}" for split, clips in SYNTHETIC_SPLIT_CLIPS.items()
    )
    config_path.write_text(
        "[frame_steps]\ntrain_tuning = 5\ntrain_final = 2\nval_monitoring = 5\n"
        f"val_reporting = 1\n\n[splits]\n{splits}\n"
    )
    return config_path


@pytest.fixture
def run_dir(tmp_path: Path, synthetic_annotations: AnnotationTables) -> Path:
    run_dir = tmp_path / "runs" / "faster_rcnn-baseline" / "20261105-1200"
    run_dir.mkdir(parents=True)
    val_objects = synthetic_annotations.objects.filter(pl.col("split") == "val")
    val_objects.select(
        "image_id", "class_id", pl.lit(0.9).alias("score"), pl.col(BOX_COLUMNS).cast(pl.Float64)
    ).write_parquet(run_dir / PREDICTIONS_FILE)
    write_metrics(run_dir, {"training": TRAINING, "inference": INFERENCE})
    run_config = load_experiment_config(FASTER_RCNN_BASELINE)
    (run_dir / CONFIG_FILE).write_text(format_toml(run_config))
    return run_dir


def test_evaluate_run_keeps_existing_metrics_and_prints_the_results_row(
    run_dir: Path, synthetic_root: Path, synthetic_data_config: Path
) -> None:
    metrics = evaluate_run(run_dir, synthetic_root, synthetic_data_config)
    assert json.loads((run_dir / METRICS_FILE).read_text()) == metrics
    assert metrics["training"] == TRAINING
    assert metrics["inference"] == INFERENCE
    assert metrics["evaluation"]["map"] == 1.0
    assert metrics["evaluation"]["frames"] == 3
    row = pl.read_csv(io.StringIO(format_results_row(run_dir, metrics)))
    assert tuple(row.columns) == RESULTS_COLUMNS
    assert row.height == 1
    values = row.row(0, named=True)
    assert values["experiment"] == "faster_rcnn-baseline"
    assert values["run_dir"] == run_dir.as_posix()
    assert values["frame_step"] == 5
    assert values["map"] == 1.0
    assert values["ms_per_image"] == 12.3456
    assert values["public_lb"] is None
    assert values["notes"] == "baseline"


def test_frames_without_rows_count_as_frames_without_detections(
    run_dir: Path, synthetic_root: Path, synthetic_data_config: Path
) -> None:
    predictions = pl.read_parquet(run_dir / PREDICTIONS_FILE)
    predictions.filter(pl.col("image_id") != 2).write_parquet(run_dir / PREDICTIONS_FILE)
    evaluation = evaluate_run(run_dir, synthetic_root, synthetic_data_config)["evaluation"]
    assert evaluation["frames"] == 3
    assert 0.0 < evaluation["map"] < 1.0


@pytest.mark.parametrize(
    "inference",
    [
        {},
        {"split": "test", "frame_step": 1, "num_frames": 3},
        {"split": "val", "frame_step": 5, "num_frames": 3},
        {"split": "val", "frame_step": 1, "num_frames": 2},
    ],
)
def test_incomplete_inference_records_are_rejected(
    run_dir: Path, synthetic_root: Path, synthetic_data_config: Path, inference: dict
) -> None:
    write_metrics(run_dir, {"training": TRAINING, "inference": inference})
    with pytest.raises(ValueError, match="evaluation needs"):
        evaluate_run(run_dir, synthetic_root, synthetic_data_config)
    assert "evaluation" not in json.loads((run_dir / METRICS_FILE).read_text())


def test_missing_training_fields_fail_after_metrics_are_written(
    run_dir: Path, synthetic_root: Path, synthetic_data_config: Path
) -> None:
    inference = {key: value for key, value in INFERENCE.items() if key != "ms_per_image"}
    write_metrics(run_dir, {"inference": inference})
    metrics = evaluate_run(run_dir, synthetic_root, synthetic_data_config)
    assert "evaluation" in json.loads((run_dir / METRICS_FILE).read_text())
    with pytest.raises(ValueError, match=r"training\.seed.*inference\.ms_per_image"):
        format_results_row(run_dir, metrics)


def test_predictions_for_unknown_frames_are_rejected(
    run_dir: Path, synthetic_root: Path, synthetic_data_config: Path
) -> None:
    predictions = pl.read_parquet(run_dir / PREDICTIONS_FILE)
    unknown = predictions.head(1).with_columns(pl.col("image_id") + 100)
    pl.concat([predictions, unknown]).write_parquet(run_dir / PREDICTIONS_FILE)
    with pytest.raises(ValueError, match="unknown image_id"):
        evaluate_run(run_dir, synthetic_root, synthetic_data_config)
    assert "evaluation" not in json.loads((run_dir / METRICS_FILE).read_text())
