import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork

import src.training.torchvision_loop
from src.data.annotations import AnnotationTables
from src.data.dataset import VisDroneDataset
from src.evaluate import METRICS_FILE, TRAINING_FIELDS
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.torchvision_detectors import MODEL_STATE_KEY, predict_torchvision_frames
from src.models.faster_rcnn import build_faster_rcnn
from src.models.torchvision_labels import (
    shift_detections_from_torchvision,
    shift_targets_to_torchvision,
)
from src.training.experiment import init_wandb, load_experiment_config
from src.training.torchvision_loop import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    build_optimizer,
    check_training_budget,
    collate_detection_batch,
    train_torchvision_detector,
)
from tests.conftest import SYNTHETIC_EMPTY_FRAME, SYNTHETIC_SPLIT_CLIPS
from tests.test_smoke import FASTER_RCNN_BASELINE

FASTER_RCNN_CONFIG_DIR = FASTER_RCNN_BASELINE.parent
TINY_INPUT = {"min_size": 64, "max_size": 96}
FRAME_WIDTH, FRAME_HEIGHT = 64, 48


@pytest.fixture(scope="module")
def baseline_config() -> dict:
    return load_experiment_config(FASTER_RCNN_BASELINE)


@pytest.fixture(scope="module")
def baseline_model(baseline_config: dict) -> FasterRCNN:
    return build_faster_rcnn(baseline_config)


def test_target_labels_are_shifted_up_and_other_keys_dropped() -> None:
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    targets = [
        {"boxes": boxes, "labels": torch.tensor([0, 7]), "ignored_boxes": torch.zeros((1, 4))},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)},
    ]
    shifted = shift_targets_to_torchvision(targets)
    assert [set(target) for target in shifted] == [{"boxes", "labels"}] * 2
    assert shifted[0]["labels"].tolist() == [1, 8] and shifted[0]["boxes"] is boxes
    assert shifted[1]["labels"].shape == (0,)
    assert targets[0]["labels"].tolist() == [0, 7]


def test_detection_labels_are_shifted_back_to_0_based() -> None:
    detections = [
        {"boxes": torch.ones((2, 4)), "scores": torch.ones(2), "labels": torch.tensor([1, 8])}
    ]
    [shifted] = shift_detections_from_torchvision(detections)
    assert shifted["labels"].tolist() == [0, 7]
    assert shifted["boxes"] is detections[0]["boxes"]
    assert detections[0]["labels"].tolist() == [1, 8]


def test_baseline_keeps_torchvision_defaults_except_inference_and_classes(
    baseline_model: FasterRCNN,
) -> None:
    default = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    assert baseline_model.transform.min_size == (1024,)
    assert baseline_model.transform.max_size == 1824
    rpn, default_rpn = baseline_model.rpn, default.rpn
    heads, default_heads = baseline_model.roi_heads, default.roi_heads
    assert isinstance(rpn, RegionProposalNetwork) and isinstance(default_rpn, RegionProposalNetwork)
    assert isinstance(heads, RoIHeads) and isinstance(default_heads, RoIHeads)
    assert rpn.anchor_generator.sizes == default_rpn.anchor_generator.sizes
    assert rpn.anchor_generator.aspect_ratios == default_rpn.anchor_generator.aspect_ratios
    assert rpn._pre_nms_top_n == default_rpn._pre_nms_top_n
    assert rpn._post_nms_top_n == default_rpn._post_nms_top_n
    for attribute in ("nms_thresh", "score_thresh"):
        assert getattr(rpn, attribute) == getattr(default_rpn, attribute)
    for module, default_module in ((rpn, default_rpn), (heads, default_heads)):
        matcher, default_matcher = module.proposal_matcher, default_module.proposal_matcher
        assert (matcher.high_threshold, matcher.low_threshold) == (
            default_matcher.high_threshold,
            default_matcher.low_threshold,
        )
        sampler, default_sampler = module.fg_bg_sampler, default_module.fg_bg_sampler
        assert (sampler.batch_size_per_image, sampler.positive_fraction) == (
            default_sampler.batch_size_per_image,
            default_sampler.positive_fraction,
        )
    assert (heads.score_thresh, heads.nms_thresh, heads.detections_per_img) == (0.001, 0.5, 500)
    assert heads.box_predictor.cls_score.out_features == 9
    assert heads.box_predictor.bbox_pred.out_features == 36


def test_coco_weights_are_loaded_and_the_new_predictor_is_seeded(
    baseline_config: dict, baseline_model: FasterRCNN
) -> None:
    coco = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict()
    state = baseline_model.state_dict()
    for key in (
        "rpn.head.conv.0.0.weight",
        "roi_heads.box_head.5.weight",
        "backbone.fpn.layer_blocks.0.0.weight",
    ):
        assert torch.equal(state[key], coco[key])
    predictor_key = "roi_heads.box_predictor.cls_score.weight"
    again = build_faster_rcnn(baseline_config).state_dict()[predictor_key]
    other_seed = build_faster_rcnn(baseline_config | {"seed": 7}).state_dict()[predictor_key]
    assert torch.equal(again, state[predictor_key])
    assert not torch.equal(other_seed, state[predictor_key])


def test_anchors_that_change_the_rpn_head_shape_are_rejected(baseline_config: dict) -> None:
    rpn = baseline_config["rpn"] | {"aspect_ratios": [[0.5, 1.0, 1.5, 2.0]] * 5}
    with pytest.raises(ValueError, match="Anchors per location"):
        build_faster_rcnn(baseline_config | {"rpn": rpn})


def test_frames_without_boxes_train_in_a_batch(
    baseline_config: dict, synthetic_annotations: AnnotationTables
) -> None:
    dataset = VisDroneDataset(synthetic_annotations, "train", frame_step=1)
    clip, frame_id = SYNTHETIC_EMPTY_FRAME
    empty_index = dataset.images["clip"].to_list().index(clip) + frame_id - 1
    loader = DataLoader(
        torch.utils.data.Subset(dataset, [0, empty_index]),
        batch_size=2,
        collate_fn=collate_detection_batch,
    )
    frames, targets = next(iter(loader))
    assert isinstance(frames, list) and frames[0].dtype == torch.uint8
    assert targets[1]["boxes"].shape == (0, 4) and targets[1]["labels"].shape == (0,)
    model = build_faster_rcnn(baseline_config | TINY_INPUT)
    model.train()
    losses = model([frame.float() / 255 for frame in frames], shift_targets_to_torchvision(targets))
    loss = sum(losses.values(), torch.zeros(()))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_predictions_are_0_based_top_left_pixel_boxes_with_timing(
    baseline_config: dict, tmp_path: Path
) -> None:
    frame_path = tmp_path / "frame.jpg"
    write_jpeg(torch.full((3, FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=torch.uint8), str(frame_path))
    images = pl.DataFrame(
        {"image_id": ["clip_0000001", "clip_0000002"], "file_path": [str(frame_path)] * 2}
    )
    detections = {
        "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0], [0.0, 0.0, 8.0, 6.0]]),
        "scores": torch.tensor([0.875, 0.25]),
        "labels": torch.tensor([8, 1]),
    }
    inputs = []
    model = torch.nn.Identity()

    def return_detections(
        module: torch.nn.Module, args: tuple, output: list[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        inputs.append(args[0])
        return [detections]

    model.register_forward_hook(return_detections)
    torch.save({MODEL_STATE_KEY: model.state_dict()}, tmp_path / "best.pt")
    predictions, ms_per_image = predict_torchvision_frames(
        model, tmp_path / "best.pt", images, baseline_config
    )
    expected = [[7, 0.875, 10, 5, 20, 20], [0, 0.25, 0, 0, 8, 6]]
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions["image_id"].to_list() == ["clip_0000001"] * 2 + ["clip_0000002"] * 2
    assert predictions["class_id"].dtype == pl.Int64
    np.testing.assert_allclose(
        predictions.drop("image_id").to_numpy(), expected + expected, rtol=1e-6
    )
    validate_predictions(predictions, images)
    assert ms_per_image > 0
    assert len(inputs) == baseline_config["inference"]["warmup_images"] + images.height
    [frame] = inputs[-1]
    assert frame.dtype == torch.float32 and frame.shape == (3, FRAME_HEIGHT, FRAME_WIDTH)
    assert frame.max().item() == pytest.approx(1.0)


def test_full_run_needs_epochs_batch_size_and_lr(baseline_config: dict) -> None:
    with pytest.raises(ValueError, match="epochs must be.*batch_size must be.*lr must be positive"):
        check_training_budget(baseline_config)
    timing = load_experiment_config(FASTER_RCNN_CONFIG_DIR / "timing.toml")
    check_training_budget(timing)
    assert (timing["epochs"], timing["batch_size"], timing["grad_accum"]) == (1, 4, 1)
    with pytest.raises(ValueError, match="train_frames must be"):
        check_training_budget(timing | {"train_frames": "val_reporting"})


def test_only_a_constant_lr_without_warmup_is_supported(baseline_config: dict) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(model, baseline_config | {"lr": 0.01})
    assert optimizer.param_groups[0]["momentum"] == 0.9
    for change in ({"schedule": "cosine"}, {"warmup_iters": 100}):
        with pytest.raises(ValueError, match="Only schedule 'constant'"):
            build_optimizer(model, baseline_config | change)


def test_full_run_writes_checkpoints_and_the_training_section(
    baseline_config: dict,
    synthetic_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_config = tmp_path / "data.toml"
    splits = "\n".join(
        f"{split} = {json.dumps(clips)}" for split, clips in SYNTHETIC_SPLIT_CLIPS.items()
    )
    data_config.write_text(
        f"[frame_steps]\ntrain_tuning = 1\nval_monitoring = 1\n\n[splits]\n{splits}\n"
    )
    monkeypatch.setattr(src.training.torchvision_loop, "DATA_CONFIG_PATH", data_config)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    logging_config = baseline_config["logging"] | {"log_interval": 1, "sample_frames": 1}
    budget = {"epochs": 2, "batch_size": 4, "lr": 0.001, "num_workers": 0}
    config = baseline_config | TINY_INPUT | budget | {"logging": logging_config}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wandb_run = init_wandb(config, {"entity": "entity", "project": "project"}, run_dir=None)
    train_torchvision_detector(
        build_faster_rcnn(config), config, synthetic_root, run_dir, wandb_run
    )
    wandb_run.finish()
    training = json.loads((run_dir / METRICS_FILE).read_text())["training"]
    assert set(TRAINING_FIELDS) <= set(training)
    assert (training["epochs"], training["frame_step"], training["effective_batch_size"]) == (
        2,
        1,
        4,
    )
    assert [record["epoch"] for record in training["history"]] == [1, 2]
    val_maps = [record["val/map"] for record in training["history"]]
    assert training["best_epoch"] == int(np.argmax(val_maps)) + 1
    assert {"train/loss", "train/loss_classifier", "train/loss_objectness"} <= set(
        training["history"][0]
    )
    best = torch.load(run_dir / BEST_CHECKPOINT, weights_only=True)
    last = torch.load(run_dir / LAST_CHECKPOINT, weights_only=True)
    assert (best["epoch"], last["epoch"]) == (training["best_epoch"], 2)
    build_faster_rcnn(config).load_state_dict(best[MODEL_STATE_KEY])
