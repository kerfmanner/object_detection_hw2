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
from ultralytics.utils.loss import E2ELoss

import src.inference.yolo
from src.data.annotations import AnnotationTables
from src.data.classes import CLASS_NAMES
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.yolo import convert_yolo_boxes, load_yolo_checkpoint, predict_yolo_frames
from src.models.yolo import (
    build_yolo,
    count_assignment_candidates,
    load_p2_weights,
    set_assigner_topk,
)
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


def test_flipped_frames_are_predicted_and_their_boxes_mirrored_back(
    baseline_config: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pixels = torch.zeros((3, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.uint8)
    pixels[:, :, : FRAME_WIDTH // 4] = 255
    frame_path = tmp_path / "frame.jpg"
    write_jpeg(pixels, str(frame_path))
    images = pl.DataFrame({"image_id": ["clip_0000001"], "file_path": [str(frame_path)]})
    detector = FakeDetector()
    seen: list[np.ndarray] = []
    predict = detector.predict

    def record_frame(frame: np.ndarray, **options: object) -> list[Results]:
        seen.append(frame)
        return predict(frame, **options)

    monkeypatch.setattr(detector, "predict", record_frame)
    monkeypatch.setattr(src.inference.yolo, "load_yolo_checkpoint", lambda path: detector)
    config = baseline_config | {"inference": baseline_config["inference"] | {"hflip": True}}
    predictions, _ = predict_yolo_frames(torch.nn.Identity(), tmp_path / "best.pt", images, config)
    mirrored = np.array(EXPECTED_DETECTIONS, dtype=np.float64)
    mirrored[:, 2] = FRAME_WIDTH - mirrored[:, 2] - mirrored[:, 4]
    np.testing.assert_allclose(predictions.drop("image_id").to_numpy(), mirrored)
    assert (
        seen[-1][:, -FRAME_WIDTH // 8 :].min() > 200 and seen[-1][:, : FRAME_WIDTH // 2].max() < 50
    )
    assert seen[-1].flags["C_CONTIGUOUS"]


def test_checkpoint_with_other_classes_is_rejected(baseline_config: dict) -> None:
    coco_checkpoint = build_yolo(baseline_config).ckpt_path
    assert coco_checkpoint is not None
    with pytest.raises(ValueError, match="predicts classes"):
        load_yolo_checkpoint(Path(coco_checkpoint))


def test_full_run_needs_epochs_and_batch_size(baseline_config: dict) -> None:
    with pytest.raises(ValueError, match="epochs must be.*batch_size must be"):
        check_training_budget(baseline_config | {"epochs": 0, "batch_size": 0})
    with pytest.raises(ValueError, match="clip_split must be main"):
        check_training_budget(baseline_config | {"clip_split": "fold_0"})
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


def test_assigner_topk_sets_only_the_one_to_many_branch_of_yolo26(
    smoke_annotations: AnnotationTables,
) -> None:
    detector = DetectionModel("yolo26n.yaml", nc=len(CLASS_NAMES), verbose=False)
    setattr(detector, "args", get_cfg())
    detector.train()
    set_assigner_topk(detector, 20)
    criterion = detector.criterion
    assert isinstance(criterion, E2ELoss)
    assert criterion.one2many.assigner.topk == 20
    assert (criterion.one2one.assigner.topk, criterion.one2one.assigner.topk2) == (7, 1)
    keys = smoke_annotations.images.filter(pl.col("split") == "train").head(2)
    objects = smoke_annotations.objects.join(
        keys.select("split", "image_id"), on=["split", "image_id"]
    )
    frames = AnnotationTables(images=keys, objects=objects, ignored=smoke_annotations.ignored)
    loss, _ = detector.loss(build_yolo_batch(frames))
    assert torch.isfinite(loss).all()


def test_p2_head_adds_a_stride_4_level_and_loads_the_checkpoint_weights(
    baseline_config: dict,
) -> None:
    model = build_yolo(baseline_config | {"arch": "yolo26n", "p2_head": True}).model
    plain = build_yolo(baseline_config | {"arch": "yolo26n"}).model
    assert isinstance(model, DetectionModel) and isinstance(plain, DetectionModel)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert plain.stride.tolist() == [8.0, 16.0, 32.0]
    weights, plain_weights = model.state_dict(), plain.state_dict()
    for key, plain_key in (
        ("model.0.conv.weight", "model.0.conv.weight"),
        ("model.23.conv.weight", "model.17.conv.weight"),
        ("model.28.cv1.conv.weight", "model.22.cv1.conv.weight"),
        ("model.29.cv2.1.0.conv.weight", "model.23.cv2.0.0.conv.weight"),
        ("model.29.one2one_cv3.3.2.weight", "model.23.one2one_cv3.2.2.weight"),
    ):
        assert torch.equal(weights[key], plain_weights[plain_key])
    fresh = DetectionModel("yolo26n-p2.yaml", verbose=False)
    assert load_p2_weights(fresh, plain) == len(plain_weights)


def test_small_object_recipe_trains_on_tiles_and_predicts_full_frames(
    baseline_config: dict,
) -> None:
    config = load_experiment_config(YOLO_CONFIG_DIR / "yolo26m_p2_tiles.toml")
    assert config["run"]["experiment"] == "yolo-yolo26m_p2_tiles"
    assert set(config) == set(baseline_config)
    assert config["tiles"] == baseline_config["tiles"] | {"should_tile": True}
    assert config["imgsz"] == config["tiles"]["frame_width"]
    assert config["inference"] == baseline_config["inference"]
    check_training_budget(config)


def test_large_batch_recipe_changes_only_the_batch_size() -> None:
    recipe = load_experiment_config(YOLO_CONFIG_DIR / "yolo26m_p2_tiles.toml")
    large_batch = load_experiment_config(YOLO_CONFIG_DIR / "yolo26m_p2_tiles_b16.toml")
    assert large_batch["run"]["experiment"] == "yolo-yolo26m_p2_tiles_b16"
    different = {key for key in large_batch if key != "run" and large_batch[key] != recipe[key]}
    assert {key: large_batch[key] for key in different} == {"batch_size": 16}
    check_training_budget(large_batch)


def test_step2_recipe_keeps_tiles_seen_and_warmup_steps_of_the_large_batch_recipe() -> None:
    large_batch = load_experiment_config(YOLO_CONFIG_DIR / "yolo26m_p2_tiles_b16.toml")
    step2 = load_experiment_config(YOLO_CONFIG_DIR / "yolo26m_p2_tiles_b16_step2.toml")
    different = {key for key in step2 if key != "run" and step2[key] != large_batch[key]}
    assert {key: step2[key] for key in different} == {
        "train_frames": "train_final",
        "epochs": 8,
        "warmup_epochs": 0.4,
        "close_mosaic": 2,
    }
    assert step2["warmup_epochs"] * 5 / 2 == pytest.approx(large_batch["warmup_epochs"])
    assert step2["close_mosaic"] / step2["epochs"] == pytest.approx(
        large_batch["close_mosaic"] / large_batch["epochs"]
    )
    check_training_budget(step2)


def test_assigner_topk_must_be_a_positive_integer(untrained_detector: DetectionModel) -> None:
    for topk in (0, -1):
        with pytest.raises(ValueError, match="TAL topk must be an integer"):
            set_assigner_topk(untrained_detector, topk)


def test_assignment_candidates_count_grid_points_strictly_inside_enlarged_boxes() -> None:
    boxes = np.array(
        [[0.0, 0.0, 32.0, 32.0], [16.0, 16.0, 48.0, 32.0], [100.0, 100.0, 104.0, 104.0]]
    )
    candidates, shared = count_assignment_candidates(boxes, [8, 16], min_side=16)
    assert candidates.tolist() == [16 + 4, 8 + 2, 4 + 1]
    assert shared.tolist() == [4 + 1, 4 + 1, 0]
    none, _ = count_assignment_candidates(np.zeros((0, 4)), [8], min_side=16)
    assert none.shape == (0,)
