import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.io import write_jpeg

import src.training.torchvision_loop
from src.data.annotations import AnnotationTables
from src.data.dataset import VisDroneDataset
from src.evaluate import METRICS_FILE, TRAINING_FIELDS
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.torchvision_detectors import MODEL_STATE_KEY, predict_torchvision_frames
from src.models.registry import build_model
from src.models.torchvision_labels import (
    shift_detections_from_torchvision,
    shift_targets_to_torchvision,
)
from src.training.experiment import init_wandb, load_experiment_config
from src.training.torchvision_loop import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    build_lr_scheduler,
    build_optimizer,
    check_training_budget,
    collate_detection_batch,
    freeze_batchnorm,
    select_autocast_dtype,
    train_one_epoch,
    train_torchvision_detector,
)
from tests.conftest import SYNTHETIC_EMPTY_FRAME, SYNTHETIC_SPLIT_CLIPS
from tests.test_entry_points import TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR
from tests.test_smoke import FASTER_RCNN_BASELINE, RETINANET_BASELINE, RTDETR_BASELINE

TINY_INPUT = {"min_size": 64, "max_size": 96}
FRAME_WIDTH, FRAME_HEIGHT = 64, 48
WANDB_SETTINGS = {"entity": "entity", "project": "project"}
FAMILY_LOSSES = {
    FASTER_RCNN_BASELINE: {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    },
    RETINANET_BASELINE: {"classification", "bbox_regression"},
}
TRAINABLE_BATCHNORM = "backbone.body.layer4.0.bn1"


def load_tiny_config(baseline_path: Path) -> dict:
    return load_experiment_config(baseline_path) | TINY_INPUT


@pytest.fixture
def synthetic_data_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_config = tmp_path / "data.toml"
    splits = "\n".join(
        f"{split} = {json.dumps(clips)}" for split, clips in SYNTHETIC_SPLIT_CLIPS.items()
    )
    data_config.write_text(
        f"[frame_steps]\ntrain_tuning = 1\nval_monitoring = 1\n\n[splits]\n{splits}\n"
    )
    monkeypatch.setattr(src.training.torchvision_loop, "DATA_CONFIG_PATH", data_config)
    monkeypatch.setenv("WANDB_MODE", "disabled")


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


def test_detection_labels_are_shifted_back_and_background_is_dropped() -> None:
    boxes = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    detections = [
        {"boxes": boxes, "scores": torch.tensor([0.9, 0.5, 0.1]), "labels": torch.tensor([1, 0, 8])}
    ]
    [shifted] = shift_detections_from_torchvision(detections)
    assert shifted["labels"].tolist() == [0, 7]
    assert shifted["scores"].tolist() == pytest.approx([0.9, 0.1])
    assert torch.equal(shifted["boxes"], boxes[[0, 2]])
    assert detections[0]["labels"].tolist() == [1, 0, 8]


@pytest.mark.parametrize("baseline_path", [FASTER_RCNN_BASELINE, RETINANET_BASELINE])
def test_frames_without_boxes_train_in_a_batch(
    baseline_path: Path, synthetic_annotations: AnnotationTables
) -> None:
    dataset = VisDroneDataset(synthetic_annotations, "train", frame_step=1)
    clip, frame_id = SYNTHETIC_EMPTY_FRAME
    empty_index = dataset.images["clip"].to_list().index(clip) + frame_id - 1
    loader = DataLoader(
        Subset(dataset, [0, empty_index]), batch_size=2, collate_fn=collate_detection_batch
    )
    frames, targets = next(iter(loader))
    assert isinstance(frames, list) and frames[0].dtype == torch.uint8
    assert targets[1]["boxes"].shape == (0, 4) and targets[1]["labels"].shape == (0,)
    model = build_model(load_tiny_config(baseline_path))
    model.train()
    losses = model([frame.float() / 255 for frame in frames], shift_targets_to_torchvision(targets))
    assert set(losses) == FAMILY_LOSSES[baseline_path]
    loss = sum(losses.values(), torch.zeros(()))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("is_frozen", [True, False])
def test_frozen_batchnorm_keeps_running_statistics_and_weights(
    is_frozen: bool, synthetic_annotations: AnnotationTables, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    config = load_tiny_config(FASTER_RCNN_BASELINE) | {"lr": 0.01, "freeze_batchnorm": is_frozen}
    model = build_model(config)
    if is_frozen:
        freeze_batchnorm(model)
    batchnorm = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    }
    batchnorm_keys = {
        f"{name}.{key}" for name, module in batchnorm.items() for key in module.state_dict()
    }
    before = {key: value.clone() for key, value in model.state_dict().items()}
    dataset = VisDroneDataset(synthetic_annotations, "train", frame_step=1)
    loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2, collate_fn=collate_detection_batch)
    optimizer = build_optimizer(model, config)
    scheduler = build_lr_scheduler(optimizer, "constant", warmup_iters=0, total_steps=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    wandb_run = init_wandb(config, WANDB_SETTINGS, run_dir=None)
    device = torch.device("cpu")
    train_one_epoch(model, loader, optimizer, scheduler, scaler, config, device, wandb_run, 0)
    wandb_run.finish()
    after = model.state_dict()
    changed = {key for key in after if not torch.equal(before[key], after[key])}
    assert batchnorm and changed - batchnorm_keys
    if is_frozen:
        assert not changed & batchnorm_keys
        assert not any(module.training for module in batchnorm.values())
        assert not any(
            p.requires_grad for module in batchnorm.values() for p in module.parameters()
        )
    else:
        assert f"{TRAINABLE_BATCHNORM}.running_mean" in changed
        assert f"{TRAINABLE_BATCHNORM}.weight" in changed


def test_predictions_are_0_based_top_left_pixel_boxes_with_timing(tmp_path: Path) -> None:
    config = load_experiment_config(FASTER_RCNN_BASELINE)
    frame_path = tmp_path / "frame.jpg"
    write_jpeg(torch.full((3, FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=torch.uint8), str(frame_path))
    images = pl.DataFrame(
        {"image_id": ["clip_0000001", "clip_0000002"], "file_path": [str(frame_path)] * 2}
    )
    detections = {
        "boxes": torch.tensor(
            [
                [10.0, 5.0, 30.0, 25.0],
                [1.0, 1.0, 9.0, 9.0],
                [64.0, 0.0, 64.0, 6.0],
                [0.0, 0.0, 8.0, 6.0],
            ]
        ),
        "scores": torch.tensor([0.875, 0.75, 0.5, 0.25]),
        "labels": torch.tensor([8, 0, 3, 1]),
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
        model, tmp_path / "best.pt", images, config
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
    assert len(inputs) == config["inference"]["warmup_images"] + images.height
    [frame] = inputs[-1]
    assert frame.dtype == torch.float32 and frame.shape == (3, FRAME_HEIGHT, FRAME_WIDTH)
    assert frame.max().item() == pytest.approx(1.0)


@pytest.mark.parametrize("baseline_path", [FASTER_RCNN_BASELINE, RETINANET_BASELINE])
def test_full_run_needs_epochs_batch_size_and_lr(baseline_path: Path) -> None:
    baseline = load_experiment_config(baseline_path)
    check_training_budget(baseline)
    assert (baseline["batch_size"], baseline["grad_accum"]) == (4, 1)
    with pytest.raises(ValueError, match="epochs must be.*batch_size must be.*lr must be positive"):
        check_training_budget(baseline | {"epochs": 0, "batch_size": 0, "lr": 0.0})
    timing = load_experiment_config(baseline_path.parent / "timing.toml")
    check_training_budget(timing)
    assert (timing["epochs"], timing["batch_size"], timing["grad_accum"]) == (1, 4, 1)
    assert timing["freeze_batchnorm"]
    with pytest.raises(ValueError, match="train_frames must be"):
        check_training_budget(timing | {"train_frames": "val_reporting"})


def test_only_constant_and_cosine_schedules_without_warmup_are_supported() -> None:
    config = load_experiment_config(FASTER_RCNN_BASELINE)
    model = torch.nn.Linear(1, 1)
    for schedule in ("constant", "cosine"):
        optimizer = build_optimizer(model, config | {"lr": 0.01, "schedule": schedule})
        assert optimizer.param_groups[0]["momentum"] == 0.9
    for change in ({"schedule": "step"}, {"warmup_iters": 100}):
        with pytest.raises(ValueError, match="Only schedules"):
            build_optimizer(model, config | change)


def test_cosine_schedule_decays_from_the_lr_to_zero_after_the_last_step() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(
        [{"params": [model.weight], "lr": 0.1}, {"params": [model.bias], "lr": 0.01}]
    )
    scheduler = build_lr_scheduler(optimizer, "cosine", warmup_iters=0, total_steps=4)
    rates = []
    for _ in range(4):
        rates.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    factors = [(1 + np.cos(np.pi * step / 4)) / 2 for step in range(4)]
    assert np.allclose(rates, [[0.1 * factor, 0.01 * factor] for factor in factors])
    assert factors[0] == 1 and np.all(np.diff(factors) < 0)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.0, 0.0])


def test_warmup_scales_the_lr_linearly_then_keeps_it_constant() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(
        [{"params": [model.weight], "lr": 0.1}, {"params": [model.bias], "lr": 0.01}]
    )
    scheduler = build_lr_scheduler(optimizer, "constant", warmup_iters=4, total_steps=6)
    rates = []
    for _ in range(6):
        rates.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    expected = [[0.1 * factor, 0.01 * factor] for factor in (0.25, 0.5, 0.75, 1, 1, 1)]
    assert np.allclose(rates, expected)
    constant = torch.optim.SGD(model.parameters(), lr=0.1)
    no_warmup = build_lr_scheduler(constant, "constant", warmup_iters=0, total_steps=3)
    for _ in range(3):
        assert constant.param_groups[0]["lr"] == 0.1
        constant.step()
        no_warmup.step()


def test_autocast_uses_bfloat16_only_on_gpus_with_native_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rtdetr, faster_rcnn = (
        load_experiment_config(p) for p in (RTDETR_BASELINE, FASTER_RCNN_BASELINE)
    )
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation: True)
    assert select_autocast_dtype(rtdetr, cuda) == torch.bfloat16
    assert select_autocast_dtype(rtdetr, cpu) is None
    assert select_autocast_dtype(rtdetr | {"amp": False}, cuda) is None
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda including_emulation: including_emulation
    )
    assert select_autocast_dtype(rtdetr, cuda) is None
    assert select_autocast_dtype(faster_rcnn, cuda) == torch.float16
    with pytest.raises(ValueError, match="amp_dtype must be one of"):
        select_autocast_dtype(rtdetr | {"amp_dtype": "float8"}, cuda)


@pytest.mark.parametrize("baseline_path", [FASTER_RCNN_BASELINE, RETINANET_BASELINE])
def test_full_run_writes_checkpoints_and_the_training_section(
    baseline_path: Path, synthetic_root: Path, synthetic_data_config: None, tmp_path: Path
) -> None:
    tiny_config = load_tiny_config(baseline_path)
    logging_config = tiny_config["logging"] | {"log_interval": 1, "sample_frames": 1}
    budget = {"epochs": 2, "batch_size": 4, "lr": 0.001, "num_workers": 0}
    config = tiny_config | budget | {"logging": logging_config}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wandb_run = init_wandb(config, WANDB_SETTINGS, run_dir=None)
    train_torchvision_detector(build_model(config), config, synthetic_root, run_dir, wandb_run)
    wandb_run.finish()
    training = json.loads((run_dir / METRICS_FILE).read_text())["training"]
    assert set(TRAINING_FIELDS) <= set(training)
    summary = (training["epochs"], training["frame_step"], training["effective_batch_size"])
    assert summary == (2, 1, 4) and training["precision"] == "float32"
    assert [record["epoch"] for record in training["history"]] == [1, 2]
    val_maps = [record["val/map"] for record in training["history"]]
    assert training["best_epoch"] == int(np.argmax(val_maps)) + 1
    loss_keys = {f"train/{name}" for name in FAMILY_LOSSES[baseline_path] | {"loss"}}
    assert loss_keys <= set(training["history"][0])
    best = torch.load(run_dir / BEST_CHECKPOINT, weights_only=True)
    last = torch.load(run_dir / LAST_CHECKPOINT, weights_only=True)
    assert (best["epoch"], last["epoch"]) == (training["best_epoch"], 2)
    fresh = build_model(config)
    fresh_state = fresh.state_dict()
    for key in (f"{TRAINABLE_BATCHNORM}.running_mean", f"{TRAINABLE_BATCHNORM}.weight"):
        assert torch.equal(last[MODEL_STATE_KEY][key], fresh_state[key])
    fresh.load_state_dict(best[MODEL_STATE_KEY])


def test_cosine_run_steps_the_lr_per_optimizer_step_to_zero_at_the_last_epoch(
    synthetic_root: Path,
    synthetic_annotations: AnnotationTables,
    synthetic_data_config: None,
    tmp_path: Path,
) -> None:
    tiny_config = load_tiny_config(FASTER_RCNN_BASELINE)
    logging_config = tiny_config["logging"] | {"log_interval": 1, "sample_frames": 1}
    budget = {"epochs": 2, "batch_size": 4, "grad_accum": 2, "lr": 0.001, "num_workers": 0}
    config = tiny_config | budget | {"schedule": "cosine", "logging": logging_config}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wandb_run = init_wandb(config, WANDB_SETTINGS, run_dir=None)
    train_torchvision_detector(build_model(config), config, synthetic_root, run_dir, wandb_run)
    wandb_run.finish()
    history = json.loads((run_dir / METRICS_FILE).read_text())["training"]["history"]
    batches_per_epoch = math.ceil(len(VisDroneDataset(synthetic_annotations, "train", 1)) / 4)
    steps_per_epoch = math.ceil(batches_per_epoch / 2)
    assert (batches_per_epoch, steps_per_epoch) == (3, 2)
    epoch_end_rates = [record["train/lr"] for record in history]
    halfway = (1 + np.cos(np.pi * steps_per_epoch / (2 * steps_per_epoch))) / 2
    assert epoch_end_rates == pytest.approx([0.001 * halfway, 0.0])


@pytest.mark.parametrize("field", TRAINING_ONLY_FIELDS_ADDED_WITH_RTDETR)
def test_training_requires_amp_dtype_and_grad_clip_norm_before_any_checkpoint(
    field: str, synthetic_root: Path, synthetic_data_config: None, tmp_path: Path
) -> None:
    config = load_tiny_config(FASTER_RCNN_BASELINE) | {"epochs": 1, "num_workers": 0}
    del config[field]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wandb_run = init_wandb(config, WANDB_SETTINGS, run_dir=None)
    with pytest.raises(KeyError, match=field):
        train_torchvision_detector(build_model(config), config, synthetic_root, run_dir, wandb_run)
    wandb_run.finish()
    assert not any(run_dir.iterdir())
