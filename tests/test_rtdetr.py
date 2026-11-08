import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import polars as pl
import pytest
import torch
from torchvision.io import write_jpeg
from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection

import src.training.torchvision_loop
from src.data.annotations import AnnotationTables
from src.data.classes import CLASS_NAMES
from src.data.dataset import VisDroneDataset
from src.evaluate import METRICS_FILE, TRAINING_FIELDS
from src.evaluation.coco_metrics import PREDICTION_COLUMNS, validate_predictions
from src.inference.torchvision_detectors import MODEL_STATE_KEY, predict_torchvision_frames
from src.models.registry import build_model
from src.models.rtdetr import (
    RTDetrDetector,
    build_rtdetr,
    compute_input_size,
    convert_rtdetr_outputs,
    convert_targets_to_rtdetr,
    freeze_backbone_stages,
    group_rtdetr_losses,
    predict_query_boxes,
    resize_and_pad_frames,
)
from src.training.experiment import init_wandb, load_experiment_config
from src.training.rtdetr import build_rtdetr_optimizer
from src.training.torchvision_loop import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    check_training_budget,
    train_torchvision_detector,
)
from tests.conftest import SYNTHETIC_SPLIT_CLIPS
from tests.test_smoke import RTDETR_BASELINE, RTDETR_LOSSES

RTDETR_DIR = RTDETR_BASELINE.parent
TINY_INPUT = {"min_size": 128, "max_size": 192}
WANDB_SETTINGS = {"entity": "entity", "project": "project"}
CHECKPOINT_FIELDS = (
    "num_queries",
    "num_denoising",
    "label_noise_ratio",
    "box_noise_scale",
    "matcher_class_cost",
    "matcher_bbox_cost",
    "matcher_giou_cost",
    "matcher_alpha",
    "matcher_gamma",
    "weight_loss_vfl",
    "weight_loss_bbox",
    "weight_loss_giou",
    "focal_loss_alpha",
    "focal_loss_gamma",
    "auxiliary_loss",
    "freeze_backbone_batch_norms",
)
NEW_HEAD_PREFIXES = (
    "class_embed.",
    "model.decoder.class_embed.",
    "model.enc_score_head.",
    "model.denoising_class_embed.",
)
ENCODER_BATCHNORM = "detector.model.encoder_input_proj.0.1"


@pytest.fixture(scope="module")
def baseline_config() -> dict:
    return load_experiment_config(RTDETR_BASELINE)


@pytest.fixture(scope="module")
def baseline_model(baseline_config: dict) -> RTDetrDetector:
    return build_rtdetr(baseline_config)


@pytest.fixture(scope="module")
def tiny_model(baseline_config: dict) -> RTDetrDetector:
    return build_rtdetr(baseline_config | TINY_INPUT)


@pytest.mark.parametrize(
    ("width", "height"),
    [(1920, 1080), (3840, 2160), (1904, 1071), (1360, 765), (1344, 756), (680, 382), (1400, 1050)],
)
def test_resize_keeps_the_aspect_ratio_with_the_shorter_side_at_1024(
    width: int, height: int, baseline_config: dict
) -> None:
    resized_height, resized_width = compute_input_size(
        height, width, baseline_config["min_size"], baseline_config["max_size"]
    )
    assert resized_height == 1024 and resized_width <= 1824
    assert abs(resized_width / resized_height - width / height) <= 1 / resized_height
    assert compute_input_size(width, height, 1024, 1824) == (resized_width, resized_height)


def test_the_longer_side_is_capped_at_max_size() -> None:
    assert compute_input_size(1000, 4000, 1024, 1824) == (456, 1824)


def test_frames_are_resized_and_zero_padded_to_a_multiple_of_32_without_normalization() -> None:
    frames = [torch.full((3, 48, 64), 0.25), torch.full((3, 40, 80), 0.75)]
    batch, resized_sizes = resize_and_pad_frames(frames, min_size=50, max_size=100)
    assert resized_sizes == [(50, 67), (50, 100)]
    assert batch.shape == (2, 3, 64, 128)
    assert torch.allclose(batch[0, :, :50, :67], torch.tensor(0.25))
    assert torch.allclose(batch[1, :, :50, :100], torch.tensor(0.75))
    assert batch[0, :, 50:].abs().sum() == 0 and batch[0, :, :, 67:].abs().sum() == 0
    assert batch[1, :, 50:].abs().sum() == 0 and batch[1, :, :, 100:].abs().sum() == 0


@pytest.mark.parametrize("config_name", ["baseline.toml", "timing_r18.toml"])
def test_pixel_values_follow_the_checkpoint_preprocessing(config_name: str) -> None:
    config = load_experiment_config(RTDETR_DIR / config_name)
    processor = RTDetrImageProcessor.from_pretrained(config["arch"], revision=config["revision"])
    assert processor.do_rescale and processor.rescale_factor == pytest.approx(1 / 255)
    assert not processor.do_normalize


def test_a_box_survives_the_round_trip_through_resize_and_target_conversion(
    baseline_config: dict,
) -> None:
    original_sizes = [(1080, 1920), (382, 680)]
    frames = [torch.zeros(3, *size) for size in original_sizes]
    batch, resized_sizes = resize_and_pad_frames(
        frames, baseline_config["min_size"], baseline_config["max_size"]
    )
    padded_size = (batch.shape[-2], batch.shape[-1])
    assert resized_sizes == [(1024, 1820), (1024, 1823)] and padded_size == (1024, 1824)
    boxes = [
        torch.tensor([[100.5, 200.25, 140.0, 230.0], [0.0, 0.0, 1920.0, 1080.0]]),
        torch.tensor([[10.0, 20.0, 30.0, 25.0]]),
    ]
    targets = [
        {"boxes": boxes[0], "labels": torch.tensor([0, 7]), "ignored_boxes": torch.zeros(1, 4)},
        {"boxes": boxes[1], "labels": torch.tensor([3])},
    ]
    converted = convert_targets_to_rtdetr(targets, original_sizes, resized_sizes, padded_size)
    assert [set(target) for target in converted] == [{"class_labels", "boxes"}] * 2
    assert [target["class_labels"].tolist() for target in converted] == [[0, 7], [3]]
    center_x = (100.5 + 140.0) / 2 * 1820 / 1920 / 1824
    assert converted[0]["boxes"][0, 0].item() == pytest.approx(center_x)
    assert all(((t["boxes"] >= 0) & (t["boxes"] <= 1)).all() for t in converted)
    padding_box = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    pred_boxes = torch.stack(
        [converted[0]["boxes"], torch.cat([converted[1]["boxes"], padding_box])]
    )
    logits = torch.full((2, 2, len(CLASS_NAMES)), -10.0)
    logits[0, 0, 0], logits[0, 1, 7], logits[1, 0, 3] = 5.0, 4.0, 3.0
    detections = convert_rtdetr_outputs(
        logits, pred_boxes, original_sizes, resized_sizes, padded_size, 0.5, 3
    )
    assert torch.allclose(detections[0]["boxes"], boxes[0], atol=1e-3)
    assert torch.allclose(detections[1]["boxes"], boxes[1], atol=1e-3)
    assert [frame["labels"].tolist() for frame in detections] == [[0, 7], [3]]
    assert detections[0]["scores"].tolist() == pytest.approx(
        torch.sigmoid(logits[0, [0, 1], [0, 7]])
    )


def test_outputs_rank_query_class_pairs_threshold_them_and_clip_to_the_frame() -> None:
    logits = torch.full((1, 3, len(CLASS_NAMES)), -20.0)
    logits[0, 0, 2], logits[0, 0, 5], logits[0, 1, 7], logits[0, 2, 1] = 2.0, 1.0, 0.0, -1.0
    pred_boxes = torch.tensor(
        [[[0.25, 0.25, 0.1, 0.1], [0.95, 0.5, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]]]
    )
    sizes = ([(100, 200)], [(100, 200)], (128, 224))
    [two] = convert_rtdetr_outputs(
        logits, pred_boxes, *sizes, score_thresh=0.6, detections_per_img=3
    )
    assert two["labels"].tolist() == [2, 5]
    assert torch.equal(two["boxes"][0], two["boxes"][1])
    [three] = convert_rtdetr_outputs(
        logits, pred_boxes, *sizes, score_thresh=0.4, detections_per_img=3
    )
    assert three["labels"].tolist() == [2, 5, 7]
    assert three["boxes"][2].tolist() == pytest.approx([190.4, 51.2, 200.0, 76.8])


def test_losses_are_summed_over_layers_encoder_and_denoising() -> None:
    values = {
        "loss_vfl": 1.0,
        "loss_vfl_aux_0": 2.0,
        "loss_vfl_dn_1": 3.0,
        "loss_bbox": 4.0,
        "loss_bbox_aux_5": 5.0,
        "loss_giou_dn_0": 6.0,
    }
    grouped = group_rtdetr_losses({name: torch.tensor(value) for name, value in values.items()})
    assert {name: value.item() for name, value in grouped.items()} == {
        "loss_vfl": 6.0,
        "loss_bbox": 9.0,
        "loss_giou": 6.0,
    }


def test_baseline_keeps_the_checkpoint_queries_denoising_matcher_and_losses(
    baseline_config: dict, baseline_model: RTDetrDetector
) -> None:
    checkpoint = RTDetrV2Config.from_pretrained(
        baseline_config["arch"], revision=baseline_config["revision"]
    )
    built = baseline_model.detector.config
    assert {field: getattr(built, field) for field in CHECKPOINT_FIELDS} == {
        field: getattr(checkpoint, field) for field in CHECKPOINT_FIELDS
    }
    assert built.num_labels == len(CLASS_NAMES) and built.id2label == dict(enumerate(CLASS_NAMES))
    assert baseline_model.detector.model.enc_score_head.out_features == len(CLASS_NAMES)
    assert all(
        head.out_features == len(CLASS_NAMES) for head in baseline_model.detector.class_embed
    )
    settings = (
        baseline_model.min_size,
        baseline_model.max_size,
        baseline_model.score_thresh,
        baseline_model.detections_per_img,
    )
    assert settings == (1024, 1824, 0.001, 300)


def test_coco_weights_are_loaded_and_the_new_heads_are_seeded(
    baseline_config: dict, baseline_model: RTDetrDetector
) -> None:
    coco = RTDetrV2ForObjectDetection.from_pretrained(
        baseline_config["arch"], revision=baseline_config["revision"]
    ).state_dict()
    state = baseline_model.detector.state_dict()
    mismatched = {key for key in coco if coco[key].shape != state[key].shape}
    assert mismatched and all(key.startswith(NEW_HEAD_PREFIXES) for key in mismatched)
    assert all(torch.equal(state[key], coco[key]) for key in coco.keys() - mismatched)
    head = "model.enc_score_head.weight"
    again = build_rtdetr(baseline_config).detector.state_dict()[head]
    other_seed = build_rtdetr(baseline_config | {"seed": 7}).detector.state_dict()[head]
    assert torch.equal(again, state[head]) and not torch.equal(other_seed, state[head])


def test_only_the_stem_is_frozen_and_invalid_settings_are_rejected(
    baseline_config: dict, baseline_model: RTDetrDetector
) -> None:
    backbone = baseline_model.detector.model.backbone.model
    assert not any(p.requires_grad for p in backbone.embedder.parameters())
    assert all(p.requires_grad for p in backbone.encoder.parameters())
    with pytest.raises(ValueError, match="trainable_backbone_layers must be 0 to 5"):
        freeze_backbone_stages(baseline_model.detector, 6)
    inference = baseline_config["inference"]
    with pytest.raises(ValueError, match="detections_per_img must be 1 to 300"):
        build_rtdetr(baseline_config | {"inference": inference | {"detections_per_img": 301}})
    with pytest.raises(ValueError, match="RT-DETR supports arch"):
        build_rtdetr(baseline_config | {"arch": "PekingU/rtdetr_r50vd"})


def test_optimizer_follows_the_rtdetr_recipe_parameter_groups(
    baseline_config: dict, baseline_model: RTDetrDetector
) -> None:
    optimizer = build_rtdetr_optimizer(baseline_model, baseline_config)
    main, norm, backbone = optimizer.param_groups
    assert (main["lr"], main["weight_decay"], main["betas"]) == (1e-4, 1e-4, (0.9, 0.999))
    assert (norm["lr"], norm["weight_decay"]) == (1e-4, 0.0)
    assert (backbone["lr"], backbone["weight_decay"]) == (1e-5, 1e-4)
    detector = baseline_model.detector
    trainable_backbone = {id(p) for p in detector.model.backbone.parameters() if p.requires_grad}
    assert {id(p) for p in backbone["params"]} == trainable_backbone
    norm_ids = {id(p) for p in norm["params"]}
    parameters = dict(baseline_model.named_parameters())
    for layer in ("enc_output.1", "encoder_input_proj.0.1", "encoder.fpn_blocks.0.conv1.norm"):
        assert id(parameters[f"detector.model.{layer}.weight"]) in norm_ids
    trainable = {id(p) for p in baseline_model.parameters() if p.requires_grad}
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len(set(grouped)) and set(grouped) == trainable
    for change in ({"ema": True}, {"schedule": "cosine"}, {"warmup_iters": -1}):
        with pytest.raises(ValueError):
            build_rtdetr_optimizer(baseline_model, baseline_config | change)


def test_baseline_and_timing_configs_are_valid_full_runs(baseline_config: dict) -> None:
    check_training_budget(baseline_config)
    assert (baseline_config["batch_size"], baseline_config["grad_accum"]) == (2, 2)
    timing, timing_b2, timing_r18 = (
        load_experiment_config(RTDETR_DIR / name)
        for name in ("timing.toml", "timing_b2.toml", "timing_r18.toml")
    )
    for config in (timing, timing_b2, timing_r18):
        check_training_budget(config)
        assert config["epochs"] == 1
        assert config["batch_size"] * config["grad_accum"] == 4
    assert (timing_b2["batch_size"], timing_b2["grad_accum"]) == (2, 2)
    assert timing["arch"] == timing_b2["arch"] == "PekingU/rtdetr_v2_r50vd"
    r18 = build_rtdetr(timing_r18)
    assert r18.detector.config.decoder_layers == 3
    assert list(r18.detector.config.encoder_in_channels) == [128, 256, 512]


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("lr_scaled", {"lr": 2.5e-5, "backbone_lr": 2.5e-6}),
        ("denoising_groups", {"queries": {"num_denoising": 200}}),
        ("step2", {"train_frames": "train_final", "epochs": 3}),
    ],
)
def test_experiment_configs_change_one_factor_of_the_baseline(
    baseline_config: dict, name: str, changed: dict
) -> None:
    config = load_experiment_config(RTDETR_DIR / f"{name}.toml")
    assert config["run"]["experiment"] == f"rtdetr-{name}"
    assert set(config) == set(baseline_config)
    different = {key for key in config if key != "run" and config[key] != baseline_config[key]}
    changed_values = {
        key: (
            {k: v for k, v in config[key].items() if v != baseline_config[key][k]}
            if isinstance(config[key], dict)
            else config[key]
        )
        for key in different
    }
    assert changed_values == changed
    check_training_budget(config)


def test_lr_scaled_repeat_differs_from_lr_scaled_only_in_the_seed() -> None:
    scaled = load_experiment_config(RTDETR_DIR / "lr_scaled.toml")
    repeat = load_experiment_config(RTDETR_DIR / "lr_scaled_repeat.toml")
    assert repeat["run"]["experiment"] == "rtdetr-lr_scaled_repeat-s7"
    assert repeat["seed"] == 7 and scaled["seed"] == 42
    assert {k: v for k, v in repeat.items() if k not in ("run", "seed")} == {
        k: v for k, v in scaled.items() if k not in ("run", "seed")
    }
    check_training_budget(repeat)


def test_scaled_learning_rates_keep_the_backbone_ratio(
    baseline_config: dict, baseline_model: RTDetrDetector
) -> None:
    config = load_experiment_config(RTDETR_DIR / "lr_scaled.toml")
    main, norm, backbone = build_rtdetr_optimizer(baseline_model, config).param_groups
    assert (main["lr"], norm["lr"], backbone["lr"]) == (2.5e-5, 2.5e-5, 2.5e-6)
    assert backbone["lr"] / main["lr"] == pytest.approx(
        baseline_config["backbone_lr"] / baseline_config["lr"]
    )


def test_denoising_groups_config_builds_a_model_with_200_denoising_queries() -> None:
    config = load_experiment_config(RTDETR_DIR / "denoising_groups.toml")
    assert build_rtdetr(config | TINY_INPUT).detector.config.num_denoising == 200


def test_frames_without_boxes_train_in_a_batch(tiny_model: RTDetrDetector) -> None:
    frames = [torch.rand(3, 48, 64), torch.rand(3, 40, 80)]
    box_targets = {"boxes": torch.tensor([[2.0, 3.0, 12.0, 9.0]]), "labels": torch.tensor([4])}
    no_boxes = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
    tiny_model.train()
    for targets in ([box_targets, no_boxes], [no_boxes, no_boxes]):
        losses = tiny_model(frames, targets)
        assert set(losses) == RTDETR_LOSSES
        assert all(torch.isfinite(value) for value in losses.values())


class FixedRTDetrOutputs(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, pred_boxes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.register_buffer("pred_boxes", pred_boxes)
        self.input_shapes: list[tuple[int, ...]] = []

    def return_fixed_outputs(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.input_shapes.append(tuple(pixel_values.shape))
        return SimpleNamespace(logits=self.logits, pred_boxes=self.pred_boxes)

    forward = return_fixed_outputs


def test_predictions_are_0_based_top_left_pixel_boxes_with_timing(
    baseline_config: dict, tmp_path: Path
) -> None:
    frame_path = tmp_path / "frame.jpg"
    write_jpeg(torch.full((3, 48, 64), 128, dtype=torch.uint8), str(frame_path))
    images = pl.DataFrame(
        {"image_id": ["clip_0000001", "clip_0000002"], "file_path": [str(frame_path)] * 2}
    )
    logits = torch.full((1, 2, len(CLASS_NAMES)), -20.0)
    logits[0, 0, 7], logits[0, 1, 0] = 2.0, 1.0
    pred_boxes = torch.tensor([[[0.25, 0.25, 0.25, 0.125], [0.75, 0.5, 0.5, 1.0]]])
    outputs = FixedRTDetrOutputs(logits, pred_boxes)
    detector = cast(RTDetrV2ForObjectDetection, outputs)
    model = RTDetrDetector(
        detector, min_size=48, max_size=64, score_thresh=0.001, detections_per_img=3
    )
    torch.save({MODEL_STATE_KEY: model.state_dict()}, tmp_path / "best.pt")
    predictions, ms_per_image = predict_torchvision_frames(
        model, tmp_path / "best.pt", images, baseline_config
    )
    first, second = torch.sigmoid(torch.tensor([2.0, 1.0])).tolist()
    expected = [[7, first, 8, 12, 16, 8], [0, second, 32, 0, 32, 48]]
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions["image_id"].to_list() == ["clip_0000001"] * 2 + ["clip_0000002"] * 2
    assert predictions["class_id"].dtype == pl.Int64
    np.testing.assert_allclose(
        predictions.drop("image_id").to_numpy(), expected + expected, rtol=1e-5
    )
    validate_predictions(predictions, images)
    assert ms_per_image > 0
    assert len(outputs.input_shapes) == baseline_config["inference"]["warmup_images"] + 2
    assert outputs.input_shapes[-1] == (1, 3, 64, 64)


def test_full_run_writes_checkpoints_and_the_training_section(
    synthetic_root: Path,
    synthetic_annotations: AnnotationTables,
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
    baseline = load_experiment_config(RTDETR_BASELINE)
    logging_config = baseline["logging"] | {"log_interval": 1, "sample_frames": 1}
    budget = {"epochs": 2, "batch_size": 4, "grad_accum": 1, "num_workers": 0}
    config = baseline | TINY_INPUT | budget | {"logging": logging_config}
    train_frames = VisDroneDataset(synthetic_annotations, "train", frame_step=1)
    steps_per_epoch = -(-len(train_frames) // config["batch_size"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wandb_run = init_wandb(config, WANDB_SETTINGS, run_dir=None)
    train_torchvision_detector(build_model(config), config, synthetic_root, run_dir, wandb_run)
    wandb_run.finish()
    training = json.loads((run_dir / METRICS_FILE).read_text())["training"]
    assert set(TRAINING_FIELDS) <= set(training)
    summary = (training["epochs"], training["frame_step"], training["effective_batch_size"])
    assert summary == (2, 1, 4) and training["precision"] == "float32"
    history = training["history"]
    assert [record["epoch"] for record in history] == [1, 2]
    assert {f"train/{name}" for name in RTDETR_LOSSES | {"loss"}} <= set(history[0])
    assert {"val/map", "val/map50", "val/ap_small", "epoch_seconds"} <= set(history[0])
    warmup_factors = [(epoch * steps_per_epoch + 1) / config["warmup_iters"] for epoch in (1, 2)]
    lrs = [record["train/lr"] for record in history]
    assert lrs == pytest.approx([config["lr"] * factor for factor in warmup_factors])
    val_maps = [record["val/map"] for record in history]
    assert training["best_epoch"] == int(np.argmax(val_maps)) + 1
    best = torch.load(run_dir / BEST_CHECKPOINT, weights_only=True)
    last = torch.load(run_dir / LAST_CHECKPOINT, weights_only=True)
    assert (best["epoch"], last["epoch"]) == (training["best_epoch"], 2)
    fresh = build_model(config)
    fresh_state = fresh.state_dict()
    for key in (f"{ENCODER_BATCHNORM}.running_mean", f"{ENCODER_BATCHNORM}.weight"):
        assert torch.equal(last[MODEL_STATE_KEY][key], fresh_state[key])
    head = "detector.model.enc_score_head.weight"
    assert not torch.equal(last[MODEL_STATE_KEY][head], fresh_state[head])
    fresh.load_state_dict(best[MODEL_STATE_KEY])


def test_query_boxes_keep_one_row_per_query_in_score_order(tiny_model: RTDetrDetector) -> None:
    tiny_model.eval()
    frame = torch.rand(3, 60, 100)
    queries = predict_query_boxes(tiny_model, frame)
    assert queries.height == tiny_model.detector.config.num_queries
    assert queries["rank"].to_list() == list(range(queries.height))
    assert queries["score"].is_sorted(descending=True)
    assert queries["class_id"].is_between(0, len(CLASS_NAMES) - 1).all()
    assert (queries["x"] >= 0).all() and (queries["x"] + queries["w"] <= 100 + 1e-6).all()
    detections = tiny_model([frame])[0]
    best = queries.row(0, named=True)
    assert best["score"] == pytest.approx(detections["scores"][0].item(), abs=1e-6)
