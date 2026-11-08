import json
import shutil
import tomllib
from pathlib import Path

import polars as pl
import pytest

import src.postprocess
from src.data.annotations import BOX_COLUMNS, AnnotationTables, load_test_images
from src.evaluate import (
    METRICS_FILE,
    PREDICTIONS_FILE,
    RESULTS_COLUMNS,
    TRAINING_FIELDS,
    evaluate_run,
    format_results_row,
)
from src.postprocess import run_postprocess
from src.predict import TEST_PREDICTIONS_FILE
from src.submission.validate import read_submission_predictions
from src.submission.writer import write_submission
from src.training.experiment import CONFIG_FILE, load_experiment_config

POSTPROCESS_DIR = Path(__file__).resolve().parents[1] / "config" / "postprocess"
MEMBER_TRAINING = dict.fromkeys(TRAINING_FIELDS, 1) | {
    "seed": 42,
    "frame_step": 5,
    "epochs": 7,
    "effective_batch_size": 4,
    "gpu_hours": 1.5,
}
TEST_BOXES = pl.DataFrame(
    {
        "image_id": ["clip_001_0000001", "clip_001_0000002"],
        "class_id": [3, 3],
        "score": [0.8, 0.6],
        "x": [10.0, 11.0],
        "y": [5.0, 5.0],
        "w": [20.0, 20.0],
        "h": [10.0, 10.0],
    }
)


def write_member_run(
    run_dir: Path,
    val_predictions: pl.DataFrame,
    test_images: pl.DataFrame,
    training: dict,
    ms_per_image: float,
) -> None:
    run_dir.mkdir(parents=True)
    val_predictions.write_parquet(run_dir / PREDICTIONS_FILE)
    write_submission(TEST_BOXES, test_images, run_dir / TEST_PREDICTIONS_FILE)
    inference = {"split": "val", "frame_step": 1, "num_frames": 3, "ms_per_image": ms_per_image}
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training, "inference": inference}))


@pytest.fixture
def members(
    prediction_data: tuple[Path, Path], synthetic_annotations: AnnotationTables, tmp_path: Path
) -> tuple[Path, list[Path]]:
    data_root, _ = prediction_data
    test_images = load_test_images(data_root)
    ground_truth = synthetic_annotations.objects.filter(pl.col("split") == "val").select(
        "image_id", "class_id", pl.lit(0.9).alias("score"), pl.col(BOX_COLUMNS).cast(pl.Float64)
    )
    trained = tmp_path / "runs" / "toy-baseline" / "20261108-1000"
    flipped = tmp_path / "runs" / "toy-infer_hflip" / "20261108-1100"
    write_member_run(trained, ground_truth, test_images, MEMBER_TRAINING, 50.0)
    flipped_training = MEMBER_TRAINING | {"source_run": trained.as_posix()}
    rescored = ground_truth.with_columns(pl.lit(0.7).alias("score"))
    write_member_run(flipped, rescored, test_images, flipped_training, 52.0)
    results_path = tmp_path / "results.csv"
    rows = [
        dict.fromkeys(RESULTS_COLUMNS, "") | {"experiment": name, "run_dir": run_dir.as_posix()}
        for name, run_dir in (("toy-baseline", trained), ("toy-infer_hflip", flipped))
    ]
    pl.DataFrame(rows).write_csv(results_path)
    return results_path, [trained, flipped]


def write_postprocess_config(tmp_path: Path, name: str, content: str) -> Path:
    family_dir = tmp_path / "config" / "postprocess"
    family_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(POSTPROCESS_DIR / "baseline.toml", family_dir / "baseline.toml")
    config_path = family_dir / f"{name}.toml"
    config_path.write_text(content)
    return config_path


def test_fused_members_become_a_run_that_evaluates_like_a_trained_one(
    members: tuple[Path, list[Path]],
    prediction_data: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_path, member_dirs = members
    clock = iter([10.0, 10.003])
    monkeypatch.setattr(src.postprocess.time, "perf_counter", lambda: next(clock))
    config_path = write_postprocess_config(
        tmp_path,
        "ens_flip",
        '# Flip TTA\nmembers = ["toy-baseline", "toy-infer_hflip"]\nweights = [1.0, 1.0]\n'
        'fusion = "wbf"\n\n[temporal]\nenabled = true\n',
    )
    data_root, data_config = prediction_data
    run_dir = run_postprocess(config_path, data_root, data_config, tmp_path / "runs", results_path)
    assert run_dir.parent == tmp_path / "runs" / "postprocess-ens_flip"
    resolved = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    assert resolved["fusion"] == "wbf" and resolved["temporal"]["enabled"]
    metrics = evaluate_run(run_dir, data_root, data_config)
    assert metrics["evaluation"]["map"] == 1.0
    training, inference = metrics["training"], metrics["inference"]
    assert training["notes"] == "Flip TTA"
    assert training["gpu_hours"] == 1.5 and training["wandb_url"] is None
    assert (training["seed"], training["frame_step"], training["epochs"]) == (42, 5, 7)
    assert [m["run_dir"] for m in training["members"]] == [d.as_posix() for d in member_dirs]
    assert inference["ms_per_image"] == pytest.approx(50.0 + 52.0 + 1000 * 0.003 / 3)
    _, row = format_results_row(run_dir, metrics).splitlines()
    assert row.startswith(f"postprocess-ens_flip,{config_path.as_posix()},")
    fused = pl.read_parquet(run_dir / PREDICTIONS_FILE)
    assert fused.height == pl.read_parquet(member_dirs[0] / PREDICTIONS_FILE).height
    assert fused["score"].to_list() == pytest.approx([0.8] * fused.height)
    test_images = load_test_images(data_root)
    test_boxes = read_submission_predictions(run_dir / TEST_PREDICTIONS_FILE, test_images)
    assert test_boxes["score"].to_list() == pytest.approx([0.75, 0.65])
    assert test_boxes["x"].to_list() == pytest.approx([10.0, 11.0])


def test_unfused_post_processing_needs_one_member_and_one_weight_each(
    members: tuple[Path, list[Path]], prediction_data: tuple[Path, Path], tmp_path: Path
) -> None:
    results_path, _ = members
    data_root, data_config = prediction_data
    for name, content, message in (
        (
            "two",
            '# Two\nmembers = ["toy-baseline", "toy-infer_hflip"]\nweights = [1.0, 1.0]\n',
            "fusion none needs exactly one member",
        ),
        (
            "weights",
            '# Weights\nmembers = ["toy-baseline", "toy-infer_hflip"]\n',
            "needs one weight per member",
        ),
    ):
        config_path = write_postprocess_config(tmp_path, name, content)
        with pytest.raises(ValueError, match=message):
            run_postprocess(config_path, data_root, data_config, tmp_path / "runs", results_path)


@pytest.mark.parametrize("config_path", sorted(POSTPROCESS_DIR.glob("*.toml")))
def test_post_processing_configs_resolve_with_one_weight_per_member(config_path: Path) -> None:
    config = load_experiment_config(config_path)
    assert config["run"]["experiment"] == f"postprocess-{config_path.stem}"
    assert len(config["weights"]) == len(config["members"]) and min(config["weights"]) > 0
    assert (config["fusion"] == "none") == (len(config["members"]) == 1)
    assert config_path.read_text().startswith("# ")
