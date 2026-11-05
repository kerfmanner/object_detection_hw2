import json
import logging
import math
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.annotations import AnnotationTables, load_annotations, load_split_clips
from src.data.dataset import VisDroneDataset, load_frame_steps, select_frames, select_smoke_frames
from src.evaluate import DATA_CONFIG_PATH, METRICS_FILE
from src.evaluation.coco_metrics import evaluate_detections
from src.inference.frames import predict_frames
from src.inference.torchvision_detectors import (
    MODEL_STATE_KEY,
    convert_frames_to_inputs,
    detect_frames,
    select_device,
)
from src.models.registry import build_model
from src.models.torchvision_labels import shift_targets_to_torchvision
from src.submission.writer import format_submission
from src.training.experiment import REPO_ROOT, TRAIN_FRAME_STEPS, describe_run
from src.visualization.frames import plot_ground_truth_and_predictions

logger = logging.getLogger(__name__)

BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"
SUPPORTED_SCHEDULE = "constant"
MONITORED_METRICS = ("map", "map50", "map75", "ap_small")
BYTES_PER_GB = 1024**3
SMOKE_FRAMES_PER_SPLIT = 2
SMOKE_OVERRIDES = {
    "min_size": 256,
    "max_size": 448,
    "epochs": 1,
    "batch_size": SMOKE_FRAMES_PER_SPLIT,
    "grad_accum": 1,
    "num_workers": 0,
}
SMOKE_WARMUP_IMAGES = 1
SMOKE_LOGGING = {"log_interval": 1, "sample_frames": 1}
Batch = tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]


def check_training_budget(config: dict) -> None:
    """Check the training frames, epochs, batch size, accumulation, and LR of a full run.

    Args:
        config: Resolved configuration with `train_frames`, `epochs`, `batch_size`,
            `grad_accum`, and `lr`.
    """
    problems = []
    if config["train_frames"] not in TRAIN_FRAME_STEPS:
        problems.append(f"train_frames must be one of {TRAIN_FRAME_STEPS}")
    for key in ("epochs", "batch_size", "grad_accum"):
        if type(config[key]) is not int or config[key] < 1:
            problems.append(f"{key} must be an integer of at least 1")
    if not config["lr"] > 0:
        problems.append("lr must be positive")
    if problems:
        raise ValueError(f"Invalid config {config['run']['config']}: {'; '.join(problems)}")


def collate_detection_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> Batch:
    """Keep the frames and targets of a batch as lists, since frames differ in size and boxes.

    Args:
        batch: `VisDroneDataset` items, pairs of a uint8 frame and its target dictionary.

    Returns:
        The list of frames and the list of targets, in batch order.
    """
    return [frame for frame, _ in batch], [target for _, target in batch]


def freeze_batchnorm(model: torch.nn.Module) -> None:
    """Stop training the affine parameters of every BatchNorm layer.

    Args:
        model: Detector whose `BatchNorm2d` weights and biases get `requires_grad` False.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.requires_grad_(False)


def set_train_mode(model: torch.nn.Module, is_batchnorm_frozen: bool) -> None:
    """Put a detector in training mode, keeping BatchNorm layers in eval mode if frozen.

    In eval mode a BatchNorm layer normalizes with its stored running statistics and does
    not update them.

    Args:
        model: Detector to train.
        is_batchnorm_frozen: Whether `BatchNorm2d` layers stay in eval mode.
    """
    model.train()
    if is_batchnorm_frozen:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.SGD:
    """Build SGD with momentum over the trainable parameters, at a constant LR without warmup.

    Args:
        model: Detector whose frozen parameters have `requires_grad` False.
        config: Resolved configuration with `lr`, `momentum`, `weight_decay`, `schedule`,
            and `warmup_iters`.

    Returns:
        The optimizer.
    """
    if config["schedule"] != SUPPORTED_SCHEDULE or config["warmup_iters"] != 0:
        raise ValueError(
            f"Only schedule {SUPPORTED_SCHEDULE!r} with warmup_iters 0 is supported, got "
            f"{config['schedule']!r} with {config['warmup_iters']}"
        )
    return torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    device: torch.device,
    wandb_run: wandb.Run,
    iteration: int,
) -> tuple[dict[str, float], int]:
    """Train one epoch with optional mixed precision and gradient accumulation.

    Each batch loss is divided by `grad_accum`, and the gradients of that many consecutive
    batches are summed before an optimizer step; a last incomplete group also makes a step.
    Every `log_interval` batches the mean losses and the LR go to W&B under `iter/`, against
    the `iteration` (batch) counter. With `freeze_batchnorm`, BatchNorm layers stay in eval
    mode, so their running statistics do not change.

    Args:
        model: torchvision detector on `device`.
        loader: DataLoader with `collate_detection_batch` over a `VisDroneDataset`.
        optimizer: Optimizer from `build_optimizer`.
        scaler: Gradient scaler, enabled when training with mixed precision.
        config: Resolved configuration with `grad_accum`, `freeze_batchnorm`, and
            `[logging]` `log_interval`.
        device: Device of the model.
        wandb_run: W&B run for the loss curves.
        iteration: Number of batches trained before this epoch.

    Returns:
        Mean of each loss over the epoch's batches, with their sum under `loss`, and the
        number of batches trained after this epoch.
    """
    set_train_mode(model, config["freeze_batchnorm"])
    grad_accum, log_interval = config["grad_accum"], config["logging"]["log_interval"]
    epoch_sums: dict[str, float] = {}
    interval_sums: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (frames, targets) in enumerate(tqdm(loader, desc="Training", unit="batch")):
        inputs = convert_frames_to_inputs(frames, device)
        device_targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in shift_targets_to_torchvision(targets)
        ]
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            losses = model(inputs, device_targets)
        loss = sum(losses.values(), torch.zeros((), device=device))
        values = {"loss": loss.item()} | {name: value.item() for name, value in losses.items()}
        if not math.isfinite(values["loss"]):
            raise RuntimeError(f"Non-finite training loss at batch {iteration + 1}: {values}")
        scaler.scale(loss / grad_accum).backward()
        if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        iteration += 1
        for name, value in values.items():
            epoch_sums[name] = epoch_sums.get(name, 0.0) + value
            interval_sums[name] = interval_sums.get(name, 0.0) + value
        if (batch_index + 1) % log_interval == 0:
            interval_means = {
                f"iter/{name}": total / log_interval for name, total in interval_sums.items()
            }
            wandb_run.log(
                interval_means
                | {"iter/lr": optimizer.param_groups[0]["lr"], "iteration": iteration}
            )
            interval_sums = {}
    return {name: total / len(loader) for name, total in epoch_sums.items()}, iteration


def select_sample_frames(images: pl.DataFrame, count: int) -> pl.DataFrame:
    """Pick frames evenly spaced over an image table, including the first and the last.

    Args:
        images: Image table, e.g. the monitoring val frames.
        count: Number of frames to pick; fewer when the table is shorter.

    Returns:
        The picked rows in table order.
    """
    indices = np.unique(np.linspace(0, images.height - 1, count).round().astype(np.int64))
    return images[indices.tolist()]


def draw_sample_frames(
    tables: AnnotationTables,
    sample_images: pl.DataFrame,
    predictions: pl.DataFrame,
    score_thresh: float,
) -> list[wandb.Image]:
    """Draw GT boxes and predictions of at least `score_thresh` on frames for W&B.

    Args:
        tables: Annotation tables with the objects and ignored regions of the frames.
        sample_images: Image table rows of the frames to draw.
        predictions: Predictions on (at least) these frames with `PREDICTION_COLUMNS`.
        score_thresh: Minimum score of the drawn predictions.

    Returns:
        One W&B image per frame, captioned with its clip and frame number.
    """
    samples = []
    for frame in sample_images.iter_rows(named=True):
        is_frame = (pl.col("split") == frame["split"]) & (pl.col("image_id") == frame["image_id"])
        caption = f"{frame['clip']}, frame {frame['frame_id']}"
        fig = plot_ground_truth_and_predictions(
            Path(frame["file_path"]),
            tables.objects.filter(is_frame),
            tables.ignored.filter(is_frame),
            predictions.filter(
                pl.col("image_id") == frame["image_id"], pl.col("score") >= score_thresh
            ),
            f"{caption}; predictions with score at least {score_thresh}",
        )
        samples.append(wandb.Image(fig, caption=caption))
        plt.close(fig)
    return samples


def fit_torchvision_detector(
    model: torch.nn.Module,
    config: dict,
    train_dataset: VisDroneDataset,
    monitor_tables: AnnotationTables,
    checkpoint_dir: Path,
    device: torch.device,
    wandb_run: wandb.Run,
) -> tuple[list[dict], int]:
    """Train a torchvision detector, monitor it after every epoch, and save its checkpoints.

    The global RNG and the shuffling generator are seeded with `seed`. With
    `freeze_batchnorm`, BatchNorm parameters are left out of the optimizer and BatchNorm
    layers keep their pretrained running statistics. Each epoch trains
    on shuffled batches of `batch_size` frames (`train_one_epoch`), then predicts the
    monitoring frames one at a time in eval mode and scores them with the shared evaluator.
    `last.pt` is written after every epoch and `best.pt` whenever the monitoring mAP beats
    all earlier epochs (ties keep the earlier epoch); both hold `model` (the state dict) and
    `epoch`. The epoch record and a few frames with GT and predicted boxes go to W&B.

    Args:
        model: torchvision detector with background label 0.
        config: Resolved configuration with `epochs`, `batch_size`, `num_workers`, `amp`,
            `seed`, `freeze_batchnorm`, the optimizer fields of `build_optimizer`, and the
            `[logging]` table.
        train_dataset: Training frames.
        monitor_tables: Annotation tables whose `images` are the monitoring val frames.
        checkpoint_dir: Existing directory for `best.pt` and `last.pt`.
        device: Device to train on.
        wandb_run: W&B run for curves and sample frames.

    Returns:
        One record per epoch (`epoch`, mean `train/` losses, `train/lr`, `val/` monitoring
        metrics, `epoch_seconds`, and `gpu_memory_gb`, the peak allocated memory during
        training or None on CPU) and the 1-based epoch saved as `best.pt`.
    """
    torch.manual_seed(config["seed"])
    num_workers = config["num_workers"]
    loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model.to(device)
    if config["freeze_batchnorm"]:
        freeze_batchnorm(model)
    optimizer = build_optimizer(model, config)
    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=config["amp"] and is_cuda)
    logging_config = config["logging"]
    sample_images = select_sample_frames(monitor_tables.images, logging_config["sample_frames"])
    for metric, step_metric in (("iter/*", "iteration"), ("train/*", "epoch"), ("val/*", "epoch")):
        wandb_run.define_metric(metric, step_metric=step_metric)
    history, best_map, best_epoch, iteration = [], -1.0, 0, 0
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        losses, iteration = train_one_epoch(
            model, loader, optimizer, scaler, config, device, wandb_run, iteration
        )
        gpu_memory_gb = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB if is_cuda else None
        predictions, _ = detect_frames(model, monitor_tables.images, device, warmup_images=0)
        metrics = evaluate_detections(monitor_tables, "val", predictions)
        val_map = metrics["map"]
        if val_map is None:
            raise ValueError("The monitoring frames have no labelled objects")
        record = {
            "epoch": epoch,
            **{f"train/{name}": value for name, value in losses.items()},
            "train/lr": optimizer.param_groups[0]["lr"],
            **{f"val/{metric}": metrics[metric] for metric in MONITORED_METRICS},
            "epoch_seconds": time.perf_counter() - started,
            "gpu_memory_gb": gpu_memory_gb,
        }
        history.append(record)
        checkpoint = {MODEL_STATE_KEY: model.state_dict(), "epoch": epoch}
        torch.save(checkpoint, checkpoint_dir / LAST_CHECKPOINT)
        if val_map > best_map:
            best_map, best_epoch = val_map, epoch
            torch.save(checkpoint, checkpoint_dir / BEST_CHECKPOINT)
        samples = draw_sample_frames(
            monitor_tables, sample_images, predictions, logging_config["sample_score_thresh"]
        )
        wandb_run.log(record | {"val/samples": samples})
        logger.info(
            "Epoch %d/%d: train loss %.4f, val mAP %.4f (best %.4f at epoch %d)",
            epoch,
            config["epochs"],
            losses["loss"],
            val_map,
            best_map,
            best_epoch,
        )
    return history, best_epoch


def run_torchvision_smoke(config: dict, tables: AnnotationTables, wandb_run: wandb.Run) -> None:
    """Train one CPU step on two smoke frames and predict two smoke val frames.

    The detector is rebuilt with the small `min_size` and `max_size` of `SMOKE_OVERRIDES`.
    The batch holds the first two frames of the smoke train clip, so the epoch is a single
    step, followed by monitoring on the first two frames of the smoke val clip and their
    prediction from `best.pt` with one warmup pass. Checkpoints go to a temporary directory
    that is removed afterwards.

    Args:
        config: Resolved configuration.
        tables: Annotation tables of all configured splits.
        wandb_run: Disabled W&B run.
    """
    smoke_config = config | SMOKE_OVERRIDES
    smoke_config |= {
        "inference": config["inference"] | {"warmup_images": SMOKE_WARMUP_IMAGES},
        "logging": config["logging"] | SMOKE_LOGGING,
    }
    model = build_model(smoke_config)
    smoke_images = (
        select_smoke_frames(tables.images, DATA_CONFIG_PATH)
        .sort("split", "frame_id")
        .group_by("split", maintain_order=True)
        .head(SMOKE_FRAMES_PER_SPLIT)
    )
    val_frames = smoke_images.filter(pl.col("split") == "val")
    train_dataset = VisDroneDataset(tables._replace(images=smoke_images), "train", frame_step=1)
    with tempfile.TemporaryDirectory() as work_dir:
        checkpoint_dir = Path(work_dir)
        fit_torchvision_detector(
            model,
            smoke_config,
            train_dataset,
            tables._replace(images=val_frames),
            checkpoint_dir,
            torch.device("cpu"),
            wandb_run,
        )
        predictions, ms_per_image = predict_frames(
            model, checkpoint_dir / BEST_CHECKPOINT, val_frames, smoke_config
        )
    submission = format_submission(predictions, val_frames)
    logger.info(
        "Smoke run predicted %d boxes on %d frames (%.0f ms/image)",
        predictions.height,
        submission.height,
        ms_per_image,
    )


def train_torchvision_detector(
    model: torch.nn.Module,
    config: dict,
    data_root: Path,
    run_dir: Path | None,
    wandb_run: wandb.Run,
) -> None:
    """Train a torchvision detector on the adopted split and record the run in its directory.

    Trains on the `train_frames` step of the train clips and monitors on the
    `val_monitoring` step of the val clips (`fit_torchvision_detector`), writing `best.pt`
    and `last.pt` to `run_dir` and the `training` section of `metrics.json`. A smoke run
    (`run_dir` None) ignores `model`, rebuilds it at the smoke input size, and writes nothing
    (`run_torchvision_smoke`).

    Args:
        model: Detector from `build_model`.
        config: Resolved configuration from `load_experiment_config`.
        data_root: Dataset root with the provided splits.
        run_dir: Run directory, or None for a smoke run.
        wandb_run: W&B run for curves and sample frames.
    """
    tables = load_annotations(data_root, load_split_clips(DATA_CONFIG_PATH))
    if run_dir is None:
        run_torchvision_smoke(config, tables, wandb_run)
        return
    check_training_budget(config)
    frame_steps = load_frame_steps(DATA_CONFIG_PATH)
    train_step, val_step = frame_steps[config["train_frames"]], frame_steps["val_monitoring"]
    train_dataset = VisDroneDataset(tables, "train", train_step)
    val_images = tables.images.filter(pl.col("split") == "val")
    monitor_tables = tables._replace(images=select_frames(val_images, val_step))
    started = time.perf_counter()
    history, best_epoch = fit_torchvision_detector(
        model, config, train_dataset, monitor_tables, run_dir, select_device(), wandb_run
    )
    gpu_hours = (time.perf_counter() - started) / 3600
    training = describe_run(config, REPO_ROOT) | {
        "frame_step": train_step,
        "val_frame_step": val_step,
        "epochs": len(history),
        "batch_size": config["batch_size"],
        "grad_accum": config["grad_accum"],
        "effective_batch_size": config["batch_size"] * config["grad_accum"],
        "gpu_hours": gpu_hours,
        "wandb_url": wandb_run.url,
        "history": history,
        "best_epoch": best_epoch,
    }
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training}, indent=2) + "\n")
    logger.info("Best epoch %d of %d; wrote %s", best_epoch, len(history), run_dir / METRICS_FILE)
