from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torchvision.io import ImageReadMode, decode_image
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

import src.training.torchvision_loop
import src.training.yolo
from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.boxes import convert_xywh_to_yolo
from src.data.classes import CLASS_NAMES
from src.data.dataset import VisDroneDataset
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.frames import predict_frames
from src.models.registry import build_model
from src.models.rtdetr import RTDetrDetector
from src.models.torchvision_labels import (
    shift_detections_from_torchvision,
    shift_targets_to_torchvision,
)
from src.models.yolo import build_yolo
from src.submission.writer import format_submission
from src.train import RUNS_ROOT, run_training
from src.training.experiment import load_experiment_config
from src.training.torchvision_loop import SMOKE_OVERRIDES
from tests.conftest import PROJECT_ROOT

YOLO_BASELINE = PROJECT_ROOT / "config" / "yolo" / "baseline.toml"
FASTER_RCNN_BASELINE = PROJECT_ROOT / "config" / "faster_rcnn" / "baseline.toml"
RETINANET_BASELINE = PROJECT_ROOT / "config" / "retinanet" / "baseline.toml"
RTDETR_BASELINE = PROJECT_ROOT / "config" / "rtdetr" / "baseline.toml"
RTDETR_LOSSES = {"loss_vfl", "loss_bbox", "loss_giou"}
TORCHVISION_BASELINES = [FASTER_RCNN_BASELINE, RETINANET_BASELINE]
SMOKE_INPUT_SIZE = 320
YOLO_OPTIMIZER_FIELDS = ("optimizer", "lr", "momentum", "warmup_bias_lr", "close_mosaic")


def build_yolo_batch(frames: AnnotationTables) -> dict[str, torch.Tensor]:
    images, batch_idx, classes, boxes = [], [], [], []
    for index, frame in enumerate(frames.images.iter_rows(named=True)):
        image = decode_image(frame["file_path"], mode=ImageReadMode.RGB).float() / 255
        size = (SMOKE_INPUT_SIZE, SMOKE_INPUT_SIZE)
        images.append(torch.nn.functional.interpolate(image[None], size=size)[0])
        objects = frames.objects.filter(pl.col("image_id") == frame["image_id"])
        xywh = objects.select(BOX_COLUMNS).to_numpy().astype(np.float64)
        boxes.append(convert_xywh_to_yolo(xywh, frame["width"], frame["height"]))
        classes.append(objects["class_id"].to_numpy())
        batch_idx.append(np.full(objects.height, index))
    return {
        "img": torch.stack(images),
        "batch_idx": torch.from_numpy(np.concatenate(batch_idx)).float(),
        "cls": torch.from_numpy(np.concatenate(classes)).float()[:, None],
        "bboxes": torch.from_numpy(np.concatenate(boxes)).float(),
    }


def list_run_files() -> set[Path]:
    return set(RUNS_ROOT.rglob("*")) if RUNS_ROOT.exists() else set()


def test_yolo_builder_runs_a_forward_and_backward_pass_on_two_frames(
    smoke_annotations: AnnotationTables,
) -> None:
    config = load_experiment_config(YOLO_BASELINE)
    pretrained = build_yolo(config).model
    assert isinstance(pretrained, torch.nn.Module)
    detector = DetectionModel(f"{config['arch']}.yaml", nc=len(CLASS_NAMES), verbose=False)
    detector.load(pretrained, verbose=False)
    setattr(detector, "args", get_cfg())
    detector.train()
    keys = smoke_annotations.images.filter(pl.col("split") == "train").head(2)
    frames = AnnotationTables(
        images=keys,
        objects=smoke_annotations.objects.join(
            keys.select("split", "image_id"), on=["split", "image_id"]
        ),
        ignored=smoke_annotations.ignored,
    )
    loss, _ = detector.loss(build_yolo_batch(frames))
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    gradients = [p.grad for p in detector.parameters() if p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)


def test_yolo_smoke_run_writes_nothing_and_predicts_valid_submission_rows(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    calls = []

    def record_prediction(
        model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
    ) -> tuple[pl.DataFrame, float]:
        outputs = predict_frames(model, checkpoint_path, images, config)
        train_args = torch.load(checkpoint_path, weights_only=False)["train_args"]
        calls.append(
            {"images": images, "config": config, "outputs": outputs, "train_args": train_args}
        )
        return outputs

    monkeypatch.setattr(src.training.yolo, "predict_frames", record_prediction)
    runs_before = list_run_files()
    assert run_training(YOLO_BASELINE, data_root, tmp_path / "runs", is_smoke=True) is None
    assert not (tmp_path / "runs").exists()
    assert list_run_files() == runs_before
    assert len(calls) == 1
    images, config = calls[0]["images"], calls[0]["config"]
    predictions, _ = calls[0]["outputs"]
    assert images.height == 2 and images["split"].unique().to_list() == ["val"]
    assert config["imgsz"] == SMOKE_INPUT_SIZE and config["inference"]["warmup_images"] == 1
    train_args = calls[0]["train_args"]
    assert {
        "optimizer": train_args["optimizer"],
        "lr": train_args["lr0"],
        "momentum": train_args["momentum"],
        "warmup_bias_lr": train_args["warmup_bias_lr"],
        "close_mosaic": train_args["close_mosaic"],
    } == {key: config[key] for key in YOLO_OPTIMIZER_FIELDS}
    assert predictions.columns == list(PREDICTION_COLUMNS) and predictions.height > 0
    validate_predictions(predictions, images)
    submission = format_submission(predictions, images)
    assert submission["image_id"].to_list() == images["image_id"].to_list()
    assert (submission["PredictionString"] != "").all()


@pytest.mark.parametrize("baseline_path", TORCHVISION_BASELINES)
def test_torchvision_builder_runs_a_forward_and_backward_pass_on_two_frames(
    baseline_path: Path,
    smoke_annotations: AnnotationTables,
) -> None:
    config = load_experiment_config(baseline_path)
    model = build_model(
        config | {"min_size": SMOKE_OVERRIDES["min_size"], "max_size": SMOKE_OVERRIDES["max_size"]}
    )
    dataset = VisDroneDataset(smoke_annotations, "train", frame_step=1)
    frames, targets = zip(*(dataset[i] for i in range(2)), strict=True)
    inputs = [frame.float() / 255 for frame in frames]
    assert 0 in torch.cat([target["labels"] for target in targets]).tolist()
    model.train()
    losses = model(inputs, shift_targets_to_torchvision(list(targets)))
    loss = sum(losses.values(), torch.zeros(()))
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)
    model.eval()
    with torch.inference_mode():
        raw = model(inputs)
    labels = torch.cat([d["labels"] for d in shift_detections_from_torchvision(raw)])
    assert labels.numel() > 0 and labels.min() >= 0 and labels.max() <= len(CLASS_NAMES) - 1


@pytest.mark.parametrize("baseline_path", TORCHVISION_BASELINES)
def test_torchvision_smoke_run_writes_nothing_and_predicts_valid_submission_rows(
    baseline_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    calls = []

    def record_prediction(
        model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
    ) -> tuple[pl.DataFrame, float]:
        outputs = predict_frames(model, checkpoint_path, images, config)
        calls.append({"images": images, "config": config, "outputs": outputs, "model": model})
        return outputs

    monkeypatch.setattr(src.training.torchvision_loop, "predict_frames", record_prediction)
    runs_before = list_run_files()
    assert run_training(baseline_path, data_root, tmp_path / "runs", is_smoke=True) is None
    assert not (tmp_path / "runs").exists()
    assert list_run_files() == runs_before
    assert len(calls) == 1
    images, config, model = calls[0]["images"], calls[0]["config"], calls[0]["model"]
    predictions, _ = calls[0]["outputs"]
    assert images.height == 2 and images["split"].unique().to_list() == ["val"]
    assert (config["min_size"], config["max_size"]) == (256, 448)
    assert model.transform.min_size == (256,) and model.transform.max_size == 448
    assert config["inference"]["warmup_images"] == 1
    assert predictions.columns == list(PREDICTION_COLUMNS) and predictions.height > 0
    validate_predictions(predictions, images)
    submission = format_submission(predictions, images)
    assert submission["image_id"].to_list() == images["image_id"].to_list()
    assert (submission["PredictionString"] != "").all()


@pytest.fixture(scope="module")
def rtdetr_smoke_model() -> torch.nn.Module:
    config = load_experiment_config(RTDETR_BASELINE)
    return build_model(
        config | {"min_size": SMOKE_OVERRIDES["min_size"], "max_size": SMOKE_OVERRIDES["max_size"]}
    )


def load_two_smoke_train_frames(
    smoke_annotations: AnnotationTables,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    dataset = VisDroneDataset(smoke_annotations, "train", frame_step=1)
    frames, targets = zip(*(dataset[i] for i in range(2)), strict=True)
    return [frame.float() / 255 for frame in frames], list(targets)


def test_rtdetr_builder_runs_a_forward_and_backward_pass_on_two_frames(
    rtdetr_smoke_model: torch.nn.Module, smoke_annotations: AnnotationTables
) -> None:
    inputs, targets = load_two_smoke_train_frames(smoke_annotations)
    assert 0 in torch.cat([target["labels"] for target in targets]).tolist()
    rtdetr_smoke_model.train()
    rtdetr_smoke_model.zero_grad()
    losses = rtdetr_smoke_model(inputs, targets)
    assert set(losses) == RTDETR_LOSSES
    loss = sum(losses.values(), torch.zeros(()))
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in rtdetr_smoke_model.parameters() if p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)
    rtdetr_smoke_model.eval()
    with torch.inference_mode():
        detections = rtdetr_smoke_model(inputs)
    assert [len(frame["labels"]) for frame in detections] == [300, 300]
    labels = torch.cat([frame["labels"] for frame in detections])
    assert labels.min() >= 0 and labels.max() <= len(CLASS_NAMES) - 1


def test_rtdetr_losses_are_finite_under_bfloat16_autocast(
    rtdetr_smoke_model: torch.nn.Module, smoke_annotations: AnnotationTables
) -> None:
    inputs, targets = load_two_smoke_train_frames(smoke_annotations)
    rtdetr_smoke_model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = rtdetr_smoke_model(inputs, targets)
    assert all(torch.isfinite(value) for value in losses.values())


def test_rtdetr_smoke_run_writes_nothing_and_predicts_valid_submission_rows(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    calls = []

    def record_prediction(
        model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
    ) -> tuple[pl.DataFrame, float]:
        outputs = predict_frames(model, checkpoint_path, images, config)
        calls.append({"images": images, "config": config, "outputs": outputs, "model": model})
        return outputs

    monkeypatch.setattr(src.training.torchvision_loop, "predict_frames", record_prediction)
    runs_before = list_run_files()
    assert run_training(RTDETR_BASELINE, data_root, tmp_path / "runs", is_smoke=True) is None
    assert not (tmp_path / "runs").exists()
    assert list_run_files() == runs_before
    assert len(calls) == 1
    images, config, model = calls[0]["images"], calls[0]["config"], calls[0]["model"]
    predictions, _ = calls[0]["outputs"]
    assert images.height == 2 and images["split"].unique().to_list() == ["val"]
    assert isinstance(model, RTDetrDetector) and (model.min_size, model.max_size) == (256, 448)
    assert config["inference"]["warmup_images"] == 1
    assert predictions.columns == list(PREDICTION_COLUMNS) and predictions.height > 0
    assert predictions.group_by("image_id").len()["len"].max() <= 300
    validate_predictions(predictions, images)
    submission = format_submission(predictions, images)
    assert submission["image_id"].to_list() == images["image_id"].to_list()
    assert (submission["PredictionString"] != "").all()
