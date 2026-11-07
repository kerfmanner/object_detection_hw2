import copy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torchvision.io import write_jpeg
from ultralytics.cfg import get_cfg
from ultralytics.engine.results import Boxes, Results
from ultralytics.nn.tasks import DetectionModel

import src.inference.yolo
from src.data.annotations import AnnotationTables
from src.data.classes import CLASS_NAMES
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.yolo import convert_yolo_boxes, load_yolo_checkpoint, predict_yolo_frames
from src.models.yolo import build_yolo, set_assigner_topk
from src.training.experiment import load_experiment_config
from src.training.yolo import check_training_budget
from tests.conftest import PROJECT_ROOT
from tests.test_smoke import build_yolo_batch

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


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("imgsz_1280", {"imgsz": 1280}),
        ("tal_topk", {"assigner": {"topk": 20}}),
        ("scale_025", {"scale": 0.25}),
    ],
)
def test_experiment_configs_change_one_field_of_the_baseline(
    baseline_config: dict, name: str, changed: dict
) -> None:
    config = load_experiment_config(YOLO_CONFIG_DIR / f"{name}.toml")
    assert config["run"]["experiment"] == f"yolo-{name}"
    assert set(config) == set(baseline_config)
    different = {key for key in config if key != "run" and config[key] != baseline_config[key]}
    assert {key: config[key] for key in different} == changed
    check_training_budget(config)


@pytest.fixture(scope="module")
def untrained_detector(baseline_config: dict) -> DetectionModel:
    detector = DetectionModel(f"{baseline_config['arch']}.yaml", nc=len(CLASS_NAMES), verbose=False)
    setattr(detector, "args", get_cfg())
    return detector.train()


def test_assigner_topk_sets_the_loss_that_training_uses(
    untrained_detector: DetectionModel, smoke_annotations: AnnotationTables
) -> None:
    keys = smoke_annotations.images.filter(pl.col("split") == "train").head(2)
    objects = smoke_annotations.objects.join(
        keys.select("split", "image_id"), on=["split", "image_id"]
    )
    frames = AnnotationTables(images=keys, objects=objects, ignored=smoke_annotations.ignored)
    batch = build_yolo_batch(frames)
    default_loss, _ = copy.deepcopy(untrained_detector).loss(batch)
    losses = {}
    for topk in (1, 10):
        detector = copy.deepcopy(untrained_detector)
        set_assigner_topk(detector, topk)
        assert (detector.criterion.assigner.topk, detector.criterion.assigner.topk2) == (topk, topk)
        losses[topk], _ = detector.loss(batch)
    assert torch.equal(losses[10], default_loss)
    assert not torch.equal(losses[1], losses[10])


def test_assigner_topk_must_be_a_positive_integer(untrained_detector: DetectionModel) -> None:
    for topk in (0, -1):
        with pytest.raises(ValueError, match="TAL topk must be an integer"):
            set_assigner_topk(untrained_detector, topk)
