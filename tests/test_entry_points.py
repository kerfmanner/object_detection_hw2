import json
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
    RESULTS_COLUMNS,
    TRAINING_FIELDS,
    evaluate_run,
    format_results_row,
)
from src.evaluation.coco_metrics import PREDICTION_COLUMNS
from src.inference.torchvision_detectors import MODEL_STATE_KEY
from src.models import registry
from src.models.registry import build_model
from src.predict import (
    TEST_PREDICTIONS_FILE,
    create_inference_run,
    load_run_config,
    run_prediction,
)
from src.submission.validate import validate_submission
from src.train import run_training
from src.training.experiment import CONFIG_FILE, format_toml, load_experiment_config
from tests.test_experiment import BASELINE_TOML
from tests.test_smoke import FASTER_RCNN_BASELINE, RETINANET_BASELINE

TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR = ("amp_dtype", "grad_clip_norm")
TRAINING_ONLY_ROI_HEADS_FIELDS_ADDED_WITH_GIOU = ("box_loss", "giou_loss_weight")
TRAINING_ONLY_TABLES_ADDED_WITH_FOCAL_LOSS = ("focal_loss",)
TRAINING_ONLY_TABLES_ADDED_WITH_AUGMENTATION = ("augment", "ignored_regions")
INFERENCE_FIELDS_ADDED_WITH_FLIP = ("hflip",)


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
    run = {
        "experiment": "toy-baseline",
        "family": "toy",
        "baseline": (toy_family / "baseline.toml").as_posix(),
    }
    config = tomllib.loads(BASELINE_TOML) | {"run": run}
    (run_dir / CONFIG_FILE).write_text(format_toml(config))
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": {"seed": 42}}))
    return run_dir


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
    removed = (
        TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR
        + TRAINING_ONLY_TABLES_ADDED_WITH_FOCAL_LOSS
        + TRAINING_ONLY_TABLES_ADDED_WITH_AUGMENTATION
    )
    legacy = {key: value for key, value in resolved.items() if key not in removed}
    inference: dict = baseline["inference"]
    legacy["inference"] = {
        key: value
        for key, value in inference.items()
        if key not in INFERENCE_FIELDS_ADDED_WITH_FLIP
    }
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


def test_runs_resolved_before_an_inference_option_existed_get_its_baseline_value(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(FASTER_RCNN_BASELINE)
    legacy = config | {"inference": config["inference"] | {"score_thresh": 0.01}}
    del legacy["inference"]["hflip"]
    (tmp_path / CONFIG_FILE).write_text(format_toml(legacy))
    loaded = load_run_config(tmp_path)
    assert loaded["inference"] == config["inference"] | {"score_thresh": 0.01}
    assert loaded["inference"]["hflip"] is False


SOURCE_TRAINING = dict.fromkeys(TRAINING_FIELDS, 1) | {
    "config": "config/toy/baseline.toml",
    "commit": "abc1234",
    "notes": "toy baseline",
    "history": [{"epoch": 1}],
}


@pytest.fixture
def results_path(prediction_run: Path, tmp_path: Path) -> Path:
    (prediction_run / METRICS_FILE).write_text(json.dumps({"training": SOURCE_TRAINING}))
    (prediction_run / "best.pt").write_bytes(b"checkpoint")
    path = tmp_path / "results.csv"
    row = dict.fromkeys(RESULTS_COLUMNS, "") | {
        "experiment": "toy-baseline",
        "run_dir": prediction_run.as_posix(),
    }
    pl.DataFrame([row]).write_csv(path)
    return path


def test_an_inference_run_predicts_with_its_source_checkpoint_and_new_settings(
    toy_family: Path,
    prediction_run: Path,
    results_path: Path,
    prediction_data: tuple[Path, Path],
    synthetic_annotations: AnnotationTables,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = toy_family / "infer_low_score.toml"
    config_path.write_text(
        '# Lower score threshold\nsource = "toy-baseline"\n\n[inference]\nscore_threshold = 0.01\n'
    )
    run_dir = create_inference_run(config_path, tmp_path / "runs", results_path)
    assert run_dir.parent == tmp_path / "runs" / "toy-infer_low_score"
    assert (run_dir / "best.pt").is_symlink() and (
        run_dir / "best.pt"
    ).read_bytes() == b"checkpoint"
    assert not (run_dir / "best.pt").readlink().is_absolute()
    config = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    source_config = tomllib.loads((prediction_run / CONFIG_FILE).read_text())
    assert config["inference"] == {"score_threshold": 0.01}
    assert {k: v for k, v in config.items() if k not in ("inference", "run")} == {
        k: v for k, v in source_config.items() if k not in ("inference", "run")
    }
    assert config["run"] == source_config["run"] | {
        "experiment": "toy-infer_low_score",
        "config": config_path.as_posix(),
        "source_run": prediction_run.as_posix(),
    }
    training = json.loads((run_dir / METRICS_FILE).read_text())["training"]
    assert training["notes"] == "Lower score threshold"
    assert training["config"] == config_path.as_posix()
    assert training["history"] == SOURCE_TRAINING["history"]
    assert training["gpu_hours"] == SOURCE_TRAINING["gpu_hours"]
    assert (training["source_run"], training["source_commit"], training["source_notes"]) == (
        prediction_run.as_posix(),
        "abc1234",
        "toy baseline",
    )
    val_objects = synthetic_annotations.objects.filter(pl.col("split") == "val")
    ground_truth = val_objects.select(
        "image_id", "class_id", pl.lit(0.9).alias("score"), pl.col(BOX_COLUMNS).cast(pl.Float64)
    )

    def predict_ground_truth(
        model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
    ) -> tuple[pl.DataFrame, float]:
        assert checkpoint_path == run_dir / "best.pt"
        assert config["inference"]["score_threshold"] == 0.01
        return ground_truth, 7.0

    monkeypatch.setattr(src.predict, "predict_frames", predict_ground_truth)
    data_root, data_config = prediction_data
    run_prediction(run_dir, "val", data_root, data_config)
    metrics = evaluate_run(run_dir, data_root, data_config)
    assert metrics["evaluation"]["map"] == 1.0
    _, row = format_results_row(run_dir, metrics).splitlines()
    assert row.startswith(f"toy-infer_low_score,{config_path.as_posix()},{run_dir.as_posix()},")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('# Lower LR\nsource = "toy-baseline"\nlr = 0.001\n', "may only set"),
        ("# No source\n\n[inference]\nscore_threshold = 0.01\n", "must set 'source'"),
        ('# Unknown\nsource = "toy-other"\n', "0 rows for 'toy-other'"),
    ],
)
def test_inference_configs_change_only_inference_settings_of_a_known_run(
    toy_family: Path, results_path: Path, tmp_path: Path, content: str, message: str
) -> None:
    config_path = toy_family / "infer_bad.toml"
    config_path.write_text(content)
    with pytest.raises(ValueError, match=message):
        create_inference_run(config_path, tmp_path / "runs", results_path)
    assert not (tmp_path / "runs" / "toy-infer_bad").exists()


def test_inference_runs_need_a_trained_source_of_their_family(
    toy_family: Path, prediction_run: Path, results_path: Path, tmp_path: Path
) -> None:
    other_family = tmp_path / "config" / "other"
    other_family.mkdir()
    (other_family / "infer_x.toml").write_text('# Other family\nsource = "toy-baseline"\n')
    with pytest.raises(ValueError, match="is in family 'other'"):
        create_inference_run(other_family / "infer_x.toml", tmp_path / "runs", results_path)
    config_path = toy_family / "infer_x.toml"
    config_path.write_text('# Same settings\nsource = "toy-baseline"\n')
    run_dir = create_inference_run(config_path, tmp_path / "runs", results_path)
    results = pl.read_csv(results_path, infer_schema=False)
    chained = results.with_columns(
        pl.lit("toy-infer_x").alias("experiment"), pl.lit(run_dir.as_posix()).alias("run_dir")
    )
    pl.concat([results, chained]).write_csv(results_path)
    (toy_family / "infer_y.toml").write_text('# Chained\nsource = "toy-infer_x"\n')
    with pytest.raises(ValueError, match="must be a trained run"):
        create_inference_run(toy_family / "infer_y.toml", tmp_path / "runs", results_path)
