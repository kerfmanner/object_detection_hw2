from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torchvision.io import write_jpeg
from ultralytics.engine.results import Boxes, Results

import src.inference.yolo
from src.data.classes import CLASS_NAMES
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.yolo import convert_yolo_boxes, load_yolo_checkpoint, predict_yolo_frames
from src.models.yolo import build_yolo
from src.training.experiment import load_experiment_config
from src.training.yolo import check_training_budget
from tests.conftest import PROJECT_ROOT

YOLO_CONFIG_DIR = PROJECT_ROOT / "config" / "yolo"
FRAME_WIDTH, FRAME_HEIGHT = 64, 48
SYNTHETIC_BOXES = [
    [10.0, 5.0, 30.0, 25.0, 0.875, 7.0],
    [64.0, 10.0, 64.0, 20.0, 0.5, 2.0],
    [0.0, 0.0, 8.0, 6.0, 0.25, 0.0],
]
EXPECTED_DETECTIONS = [[7, 0.875, 10, 5, 20, 20], [0, 0.25, 0, 0, 8, 6]]


@pytest.fixture(scope="module")
def baseline_config() -> dict:
    return load_experiment_config(YOLO_CONFIG_DIR / "baseline.toml")


def test_boxes_become_class_score_and_top_left_pixel_boxes() -> None:
    boxes = Boxes(torch.tensor(SYNTHETIC_BOXES), orig_shape=(FRAME_HEIGHT, FRAME_WIDTH))
    np.testing.assert_allclose(convert_yolo_boxes(boxes), EXPECTED_DETECTIONS)


class FakeDetector:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, frame: np.ndarray, **options: object) -> list[Results]:
        self.calls.append(options)
        names = dict(enumerate(CLASS_NAMES))
        return [Results(frame, path="frame.jpg", names=names, boxes=torch.tensor(SYNTHETIC_BOXES))]


def test_predictions_on_a_synthetic_frame_keep_image_ids_classes_and_boxes(
    baseline_config: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame_path = tmp_path / "frame.jpg"
    write_jpeg(torch.zeros((3, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.uint8), str(frame_path))
    images = pl.DataFrame(
        {
            "image_id": ["clip_0000001", "clip_0000002"],
            "width": [FRAME_WIDTH] * 2,
            "height": [FRAME_HEIGHT] * 2,
            "file_path": [str(frame_path)] * 2,
        }
    )
    detector = FakeDetector()
    monkeypatch.setattr(src.inference.yolo, "load_yolo_checkpoint", lambda path: detector)
    predictions, ms_per_image = predict_yolo_frames(
        torch.nn.Identity(), tmp_path / "best.pt", images, baseline_config
    )
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions["image_id"].to_list() == ["clip_0000001"] * 2 + ["clip_0000002"] * 2
    assert predictions["class_id"].dtype == pl.Int64
    np.testing.assert_allclose(
        predictions.drop("image_id").to_numpy(), EXPECTED_DETECTIONS + EXPECTED_DETECTIONS
    )
    validate_predictions(predictions, images)
    assert ms_per_image > 0
    inference = baseline_config["inference"]
    assert len(detector.calls) == inference["warmup_images"] + images.height
    assert detector.calls[0] | {"device": None} == {
        "imgsz": 1824,
        "conf": inference["conf"],
        "iou": 0.5,
        "max_det": 500,
        "agnostic_nms": False,
        "device": None,
        "verbose": False,
    }


def test_checkpoint_with_other_classes_is_rejected(baseline_config: dict) -> None:
    coco_checkpoint = build_yolo(baseline_config).ckpt_path
    assert coco_checkpoint is not None
    with pytest.raises(ValueError, match="predicts classes"):
        load_yolo_checkpoint(Path(coco_checkpoint))


def test_full_run_needs_epochs_and_batch_size(baseline_config: dict) -> None:
    with pytest.raises(ValueError, match="epochs must be.*batch_size must be"):
        check_training_budget(baseline_config | {"epochs": 0, "batch_size": 0})
    timing = load_experiment_config(YOLO_CONFIG_DIR / "timing.toml")
    check_training_budget(timing)
    assert timing["close_mosaic"] == 0 and baseline_config["close_mosaic"] == 10
    with pytest.raises(ValueError, match="train_frames must be"):
        check_training_budget(baseline_config | {"train_frames": "val_reporting", "epochs": 1})
