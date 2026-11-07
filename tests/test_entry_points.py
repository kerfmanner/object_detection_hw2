import json
import shutil
import tomllib
from pathlib import Path

import polars as pl
import pytest
import torch
import wandb
from torchvision.io import write_jpeg

import src.predict
import src.train
from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.evaluate import (
    METRICS_FILE,
    PREDICTIONS_FILE,
    TRAINING_FIELDS,
    evaluate_run,
    format_results_row,
)
from src.evaluation.coco_metrics import PREDICTION_COLUMNS
from src.inference.torchvision_detectors import MODEL_STATE_KEY
from src.models import registry
from src.models.registry import build_model
from src.predict import TEST_PREDICTIONS_FILE, run_prediction
from src.submission.validate import validate_submission
from src.train import run_training
from src.training.experiment import CONFIG_FILE, format_toml, load_experiment_config
from tests.conftest import SYNTHETIC_SPLIT_CLIPS
from tests.test_experiment import BASELINE_TOML
from tests.test_smoke import FASTER_RCNN_BASELINE, RETINANET_BASELINE

TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR = ("amp_dtype", "grad_clip_norm")
TRAINING_ONLY_ROI_HEADS_FIELDS_ADDED_WITH_GIOU = ("box_loss", "giou_loss_weight")
TRAINING_ONLY_TABLES_ADDED_WITH_FOCAL_LOSS = ("focal_loss",)


@pytest.fixture
def toy_family(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    family_dir = tmp_path / "config" / "toy"
    family_dir.mkdir(parents=True)
    (family_dir / "baseline.toml").write_text(BASELINE_TOML)
    monkeypatch.setitem(registry.MODEL_BUILDERS, "toy", lambda config: torch.nn.Linear(1, 1))
    return family_dir


@pytest.fixture
def training_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def record_training(
        model: torch.nn.Module,
        config: dict,
        data_root: Path,
        run_dir: Path | None,
        wandb_run: wandb.Run,
    ) -> None:
        calls.append({"config": config, "run_dir": run_dir, "is_disabled": wandb_run.disabled})

    monkeypatch.setattr(src.train, "train_detector", record_training)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    return calls


def test_unregistered_family_fails_before_writing_anything(tmp_path: Path) -> None:
    family_dir = tmp_path / "config" / "unknown_family"
    family_dir.mkdir(parents=True)
    (family_dir / "baseline.toml").write_text(BASELINE_TOML)
    with pytest.raises(ValueError, match="No model builder registered for family 'unknown_family'"):
        run_training(family_dir / "baseline.toml", tmp_path, tmp_path / "runs", is_smoke=False)
    assert not (tmp_path / "runs").exists()


def test_smoke_training_writes_nothing_and_disables_wandb(
    toy_family: Path, training_calls: list[dict], tmp_path: Path
) -> None:
    run_dir = run_training(toy_family / "baseline.toml", tmp_path, tmp_path / "runs", True)
    assert run_dir is None
    assert not (tmp_path / "runs").exists()
    assert len(training_calls) == 1
    assert training_calls[0]["run_dir"] is None and training_calls[0]["is_disabled"]
    assert training_calls[0]["config"]["run"]["experiment"] == "toy-baseline"


def test_training_creates_the_run_directory_with_the_resolved_config(
    toy_family: Path, training_calls: list[dict], tmp_path: Path
) -> None:
    override = toy_family / "lr_002.toml"
    override.write_text("# Higher learning rate\nlr = 0.02\n")
    run_dir = run_training(override, tmp_path, tmp_path / "runs", is_smoke=False)
    assert run_dir is not None
    assert run_dir.parent == tmp_path / "runs" / "toy-lr_002"
    resolved = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    assert resolved == training_calls[0]["config"]
    assert resolved["lr"] == 0.02 and resolved["run"]["family"] == "toy"


@pytest.fixture
def prediction_run(toy_family: Path, tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "toy-baseline" / "20261106-0900"
    run_dir.mkdir(parents=True)
    config = tomllib.loads(BASELINE_TOML) | {"run": {"experiment": "toy-baseline", "family": "toy"}}
    (run_dir / CONFIG_FILE).write_text(format_toml(config))
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": {"seed": 42}}))
    return run_dir


@pytest.fixture
def prediction_data(
    synthetic_root: Path, synthetic_annotations: AnnotationTables, tmp_path: Path
) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    shutil.copytree(synthetic_root, data_root)
    (data_root / "test_images.csv").write_text(
        "image_id,file_name,clip,frame,width,height\n"
        "clip_001_0000001,test/images/clip_001/0000001.jpg,clip_001,1,64,48\n"
        "clip_001_0000002,test/images/clip_001/0000002.jpg,clip_001,2,64,48\n"
    )
    config_path = tmp_path / "data.toml"
    splits = "\n".join(f"{split} = {json.dumps(c)}" for split, c in SYNTHETIC_SPLIT_CLIPS.items())
    config_path.write_text(f"[splits]\n{splits}\n")
    return data_root, config_path


def test_val_prediction_writes_parquet_and_inference_record_for_evaluate(
    prediction_run: Path,
    prediction_data: tuple[Path, Path],
    synthetic_annotations: AnnotationTables,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    val_objects = synthetic_annotations.objects.filter(pl.col("split") == "val")
    ground_truth = val_objects.select(
        "image_id", "class_id", pl.lit(0.9).alias("score"), pl.col(BOX_COLUMNS).cast(pl.Float64)
    )
    seen: list[pl.DataFrame] = []

    def predict_ground_truth(
        model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
    ) -> tuple[pl.DataFrame, float]:
        seen.append(images)
        return ground_truth, 12.5

    monkeypatch.setattr(src.predict, "predict_frames", predict_ground_truth)
    data_root, data_config = prediction_data
    output = run_prediction(prediction_run, "val", data_root, data_config)
    assert output == prediction_run / PREDICTIONS_FILE
    assert seen[0]["split"].unique().to_list() == ["val"] and seen[0].height == 3
    metrics = json.loads((prediction_run / METRICS_FILE).read_text())
    assert metrics["training"] == {"seed": 42}
    assert metrics["inference"] == {
        "split": "val",
        "frame_step": 1,
        "num_frames": 3,
        "ms_per_image": 12.5,
    }
    evaluation = evaluate_run(prediction_run, data_root, data_config)["evaluation"]
    assert evaluation["map"] == 1.0


def test_test_prediction_writes_a_valid_kaggle_csv(
    prediction_run: Path, prediction_data: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def predict_one_box(
        model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
    ) -> tuple[pl.DataFrame, float]:
        assert checkpoint_path == prediction_run / "best.pt"
        return (
            images.head(1).select(
                "image_id",
                pl.lit(1).alias("class_id"),
                pl.lit(0.5).alias("score"),
                *(pl.lit(value).alias(c) for c, value in zip(BOX_COLUMNS, (60.0, 1.0, 10.0, 5.0))),
            ),
            0.0,
        )

    monkeypatch.setattr(src.predict, "predict_frames", predict_one_box)
    data_root, data_config = prediction_data
    csv_path = run_prediction(prediction_run, "test", data_root, data_config)
    assert csv_path == prediction_run / "predictions.csv"
    assert csv_path.read_text().splitlines() == [
        "image_id,PredictionString",
        "clip_001_0000001,1 0.5 60.0 1.0 4.0 5.0",
        'clip_001_0000002,""',
    ]
    test_images = pl.read_csv(data_root / "test_images.csv")
    assert validate_submission(csv_path, test_images) == 1
    assert not (prediction_run / PREDICTIONS_FILE).exists()


def test_unknown_split_is_rejected(
    prediction_run: Path, prediction_data: tuple[Path, Path]
) -> None:
    data_root, data_config = prediction_data
    with pytest.raises(ValueError, match="Unknown split 'train'"):
        run_prediction(prediction_run, "train", data_root, data_config)


@pytest.mark.parametrize("baseline_path", [FASTER_RCNN_BASELINE, RETINANET_BASELINE])
def test_runs_resolved_before_training_only_fields_were_added_still_predict_and_evaluate(
    baseline_path: Path, prediction_data: tuple[Path, Path], tmp_path: Path
) -> None:
    baseline = load_experiment_config(baseline_path)
    experiment = baseline["run"]["experiment"]
    resolved = baseline | {"min_size": 64, "max_size": 96}
    removed = TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR + TRAINING_ONLY_TABLES_ADDED_WITH_FOCAL_LOSS
    legacy = {key: value for key, value in resolved.items() if key not in removed}
    assert set(TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR) <= set(resolved) - set(legacy)
    assert ("focal_loss" in resolved) == (baseline_path == RETINANET_BASELINE)
    if "roi_heads" in baseline:
        roi_heads: dict = baseline["roi_heads"]
        legacy["roi_heads"] = {
            key: value
            for key, value in roi_heads.items()
            if key not in TRAINING_ONLY_ROI_HEADS_FIELDS_ADDED_WITH_GIOU
        }
        assert set(roi_heads) - set(legacy["roi_heads"]) == set(
            TRAINING_ONLY_ROI_HEADS_FIELDS_ADDED_WITH_GIOU
        )
    run_dir = tmp_path / "runs" / experiment / "20261106-0900"
    run_dir.mkdir(parents=True)
    (run_dir / CONFIG_FILE).write_text(format_toml(legacy))
    model = build_model(legacy)
    torch.save({MODEL_STATE_KEY: model.state_dict(), "epoch": 1}, run_dir / "best.pt")
    training = dict.fromkeys(TRAINING_FIELDS, 1)
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training}))
    data_root, data_config = prediction_data
    for frame in (1, 2):
        frame_path = data_root / "test" / "images" / "clip_001" / f"{frame:07d}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        write_jpeg(torch.full((3, 48, 64), 40 * frame, dtype=torch.uint8), str(frame_path))
    assert run_prediction(run_dir, "val", data_root, data_config) == run_dir / PREDICTIONS_FILE
    assert pl.read_parquet(run_dir / PREDICTIONS_FILE).columns == list(PREDICTION_COLUMNS)
    metrics = evaluate_run(run_dir, data_root, data_config)
    assert metrics["training"] == training and metrics["evaluation"]["map"] is not None
    _, row = format_results_row(run_dir, metrics).splitlines()
    assert row.startswith(experiment)
    csv_path = run_prediction(run_dir, "test", data_root, data_config)
    assert csv_path == run_dir / TEST_PREDICTIONS_FILE
    test_images = pl.read_csv(data_root / "test_images.csv")
    validate_submission(csv_path, test_images)
