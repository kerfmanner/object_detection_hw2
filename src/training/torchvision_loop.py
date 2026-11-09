import copy
import json
import logging
import math
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import wandb
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from tqdm.auto import tqdm

from src.data.annotations import AnnotationTables, load_annotations, load_split_clips
from src.data.augment import build_train_transform, draw_input_sizes
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
from src.models.faster_rcnn import set_box_regression_loss
from src.models.registry import build_model
from src.models.retinanet import set_focal_loss
from src.models.rtdetr import RTDetrDetector
from src.models.torchvision_labels import TORCHVISION_FAMILIES, shift_targets_to_torchvision
from src.submission.writer import format_submission
from src.training.experiment import (
    REPO_ROOT,
    TRAIN_FRAME_STEPS,
    TRAIN_SPLIT_CHOICES,
    describe_run,
)
from src.training.rtdetr import build_rtdetr_optimizer
from src.visualization.frames import plot_ground_truth_and_predictions

logger = logging.getLogger(__name__)

BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"
SUPPORTED_SCHEDULES = ("constant", "cosine")
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
AUTOCAST_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
Batch = tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]
InputSize = tuple[int, int]


def check_training_budget(config: dict) -> None:
    """Check the training frames, epochs, batch size, accumulation, LR, augmentation
    schedule, and weight averaging of a full run.

    Args:
        config: Resolved configuration with `train_frames`, `train_splits`, `epochs`,
            `batch_size`, `grad_accum`, `lr`, and the `[augment]` and `[ema]` tables.
    """
    problems = []
    if config["train_frames"] not in TRAIN_FRAME_STEPS:
        problems.append(f"train_frames must be one of {TRAIN_FRAME_STEPS}")
    if config["train_splits"] not in TRAIN_SPLIT_CHOICES:
        problems.append(f"train_splits must be one of {TRAIN_SPLIT_CHOICES}")
    for key in ("epochs", "batch_size", "grad_accum"):
        if type(config[key]) is not int or config[key] < 1:
            problems.append(f"{key} must be an integer of at least 1")
    if not config["lr"] > 0:
        problems.append("lr must be positive")
    clean_epochs = config["augment"]["clean_epochs"]
    if type(clean_epochs) is not int or not 0 <= clean_epochs < config["epochs"]:
        problems.append("augment.clean_epochs must be an integer from 0 to epochs - 1")
    ema = config["ema"]
    if not 0 < ema["decay"] < 1:
        problems.append("ema.decay must be in (0, 1)")
    if type(ema["warmup_steps"]) is not int or ema["warmup_steps"] < 1:
        problems.append("ema.warmup_steps must be an integer of at least 1")
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


def set_input_size(model: torch.nn.Module, min_size: int, max_size: int) -> None:
    """Set the shorter-side size and the longer-side bound to which a detector resizes frames.

    A torchvision detector gets a new `GeneralizedRCNNTransform` with the same normalization
    and padding; the transform has no parameters, so training is unaffected otherwise.

    Args:
        model: Wrapped RT-DETR, or a torchvision detector with a `GeneralizedRCNNTransform`.
        min_size: Shorter frame side in px.
        max_size: Upper bound of the longer frame side in px.
    """
    if isinstance(model, RTDetrDetector):
        model.min_size, model.max_size = min_size, max_size
        return
    transform = model.transform if isinstance(model, (FasterRCNN, RetinaNet)) else None
    if not isinstance(transform, GeneralizedRCNNTransform):
        raise TypeError(f"Cannot set the input size of {type(model).__name__}")
    model.transform = GeneralizedRCNNTransform(
        min_size,
        max_size,
        transform.image_mean,
        transform.image_std,
        size_divisible=transform.size_divisible,
        fixed_size=transform.fixed_size,
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.SGD:
    """Build SGD with momentum over the trainable parameters, for a schedule without warmup.

    Args:
        model: Detector whose frozen parameters have `requires_grad` False.
        config: Resolved configuration with `lr`, `momentum`, `weight_decay`, `schedule`
            (`constant` or `cosine`, applied by `build_lr_scheduler`), and `warmup_iters`.

    Returns:
        The optimizer.
    """
    if config["schedule"] not in SUPPORTED_SCHEDULES or config["warmup_iters"] != 0:
        raise ValueError(
            f"Only schedules {SUPPORTED_SCHEDULES} with warmup_iters 0 are supported, got "
            f"{config['schedule']!r} with {config['warmup_iters']}"
        )
    return torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )


OPTIMIZER_BUILDERS: dict[str, Callable[[torch.nn.Module, dict], torch.optim.Optimizer]] = {
    "faster_rcnn": build_optimizer,
    "retinanet": build_optimizer,
    "rtdetr": build_rtdetr_optimizer,
}


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, schedule: str, warmup_iters: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the LR schedule of a run: an optional linear warmup times a constant or cosine.

    Optimizer step `i` (0-based) uses the factor `min(1, (i + 1) / warmup_iters)` of each
    group's LR, as the official RT-DETRv2 `LinearWarmup`
    (https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/optim/warmup.py);
    `warmup_iters` 0 keeps that factor 1. The `cosine` schedule multiplies it by
    `(1 + cos(pi * i / total_steps)) / 2`, the cosine decay of SGDR without restarts
    (https://arxiv.org/abs/1608.03983), used for detectors in
    https://arxiv.org/abs/1902.04103: step 0 trains at the full LR, and the LR reaches 0
    after the last step.

    Args:
        optimizer: Optimizer whose groups hold their target LR.
        schedule: `constant` or `cosine`.
        warmup_iters: Number of optimizer steps of the warmup, at least 0.
        total_steps: Number of optimizer steps of the whole run, at least 1.

    Returns:
        Scheduler to step after every optimizer step.
    """
    is_cosine = schedule == "cosine"

    def compute_lr_factor(step: int) -> float:
        warmup = min(1.0, (step + 1) / warmup_iters) if warmup_iters else 1.0
        decay = (1 + math.cos(math.pi * step / total_steps)) / 2 if is_cosine else 1.0
        return warmup * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, compute_lr_factor)


def select_autocast_dtype(config: dict, device: torch.device) -> torch.dtype | None:
    """Select the mixed precision dtype of training, or None for float32.

    Mixed precision runs on CUDA only. bfloat16 needs native GPU support (not the emulation
    `torch.cuda.is_bf16_supported` counts by default); without it training stays float32.

    Args:
        config: Resolved configuration with `amp` and `amp_dtype` (`float16` or `bfloat16`).
        device: Device to train on.

    Returns:
        The autocast dtype, or None when training in float32.
    """
    if config["amp_dtype"] not in AUTOCAST_DTYPES:
        raise ValueError(
            f"amp_dtype must be one of {sorted(AUTOCAST_DTYPES)}, got {config['amp_dtype']!r}"
        )
    dtype = AUTOCAST_DTYPES[config["amp_dtype"]]
    if not config["amp"] or device.type != "cuda":
        return None
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        return None
    return dtype


def compute_ema_decay(ema: dict, step: int) -> float:
    """Return the decay of the weight average after an optimizer step, ramped up from 0.

    The decay is `decay * (1 - exp(-step / warmup_steps))`, as in ModelEMA of the official
    RT-DETRv2 code, so that the early average follows the fast-changing weights:
    https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/optim/ema.py

    Args:
        ema: The `[ema]` table with `decay` and `warmup_steps`.
        step: Number of optimizer steps taken, including this one.

    Returns:
        Weight of the previous average in the update, in [0, decay).
    """
    return ema["decay"] * (1 - math.exp(-step / ema["warmup_steps"]))


def update_ema_weights(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Move the floating-point parameters and buffers of the averaged copy toward the model.

    Each floating-point state entry becomes `decay * average + (1 - decay) * current`, as in
    ModelEMA of the official RT-DETRv2 code; other entries (counters) are copied.

    Args:
        ema_model: Averaged copy of `model`, updated in place.
        model: Trained model.
        decay: Weight of the previous average, in [0, 1).
    """
    current = model.state_dict()
    with torch.no_grad():
        for key, averaged in ema_model.state_dict().items():
            if averaged.is_floating_point():
                averaged.lerp_(current[key], 1 - decay)
            else:
                averaged.copy_(current[key])


def describe_precision(dtype: torch.dtype | None) -> str:
    """Name the training precision for logs and `metrics.json`.

    Args:
        dtype: Autocast dtype from `select_autocast_dtype`, or None.

    Returns:
        `float16`, `bfloat16`, or `float32`.
    """
    return "float32" if dtype is None else str(dtype).removeprefix("torch.")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict,
    device: torch.device,
    wandb_run: wandb.Run,
    iteration: int,
    input_sizes: list[InputSize] | None = None,
    ema_model: torch.nn.Module | None = None,
) -> tuple[dict[str, float], int]:
    """Train one epoch with optional mixed precision and gradient accumulation.

    Each batch loss is divided by `grad_accum`, and the gradients of that many consecutive
    batches are summed before an optimizer step; a last incomplete group also makes a step.
    Before a step, the gradients are clipped to the L2 norm `grad_clip_norm` (0 disables);
    after it, the scheduler steps. Target labels are shifted for torchvision families only.
    Every `log_interval` batches the mean losses and the LR of the first parameter group go
    to W&B under `iter/`, against the `iteration` (batch) counter. With `freeze_batchnorm`,
    BatchNorm layers stay in eval mode, so their running statistics do not change. With
    `input_sizes`, each batch is resized to its own size (multi-scale training) and the
    configured `min_size` and `max_size` are restored after the epoch. With `ema_model`,
    every optimizer step also updates the weight average with the decay of
    `compute_ema_decay` at the scheduler's step count.

    Args:
        model: Detector of the shared loop (torchvision or wrapped RT-DETR) on `device`.
        loader: DataLoader with `collate_detection_batch` over a `VisDroneDataset`.
        optimizer: Optimizer from the family's `OPTIMIZER_BUILDERS` entry.
        scheduler: LR scheduler from `build_lr_scheduler`.
        scaler: Gradient scaler, enabled when training in float16.
        config: Resolved configuration with `run.family`, `grad_accum`, `grad_clip_norm`,
            `amp`, `amp_dtype`, `freeze_batchnorm`, `[logging]` `log_interval`, and the
            `[ema]` table when `ema_model` is given.
        device: Device of the model.
        wandb_run: W&B run for the loss curves.
        iteration: Number of batches trained before this epoch.
        input_sizes: One (min_size, max_size) pair in px per batch from `draw_input_sizes`,
            or None to train at the model's input size.
        ema_model: Averaged copy of `model` to update after every optimizer step, or None.

    Returns:
        Mean of each loss over the epoch's batches, with their sum under `loss`, and the
        number of batches trained after this epoch.
    """
    set_train_mode(model, config["freeze_batchnorm"])
    grad_accum, log_interval = config["grad_accum"], config["logging"]["log_interval"]
    grad_clip_norm = config["grad_clip_norm"]
    autocast_dtype = select_autocast_dtype(config, device)
    has_background_label = config["run"]["family"] in TORCHVISION_FAMILIES
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    epoch_sums: dict[str, float] = {}
    interval_sums: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (frames, targets) in enumerate(tqdm(loader, desc="Training", unit="batch")):
        if input_sizes is not None:
            set_input_size(model, *input_sizes[batch_index])
        inputs = convert_frames_to_inputs(frames, device)
        model_targets = shift_targets_to_torchvision(targets) if has_background_label else targets
        device_targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in model_targets
        ]
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            losses = model(inputs, device_targets)
        loss = sum(losses.values(), torch.zeros((), device=device))
        values = {"loss": loss.item()} | {name: value.item() for name, value in losses.items()}
        if not math.isfinite(values["loss"]):
            raise RuntimeError(f"Non-finite training loss at batch {iteration + 1}: {values}")
        scaler.scale(loss / grad_accum).backward()
        if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(loader):
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                decay = compute_ema_decay(config["ema"], scheduler.last_epoch)
                update_ema_weights(ema_model, model, decay)
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
    if input_sizes is not None:
        set_input_size(model, config["min_size"], config["max_size"])
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


def build_train_dataset(
    tables: AnnotationTables, config: dict, frame_step: int, is_clean: bool
) -> Dataset:
    """Build the training frames of the `train_splits` clips at one frame step.

    Args:
        tables: Annotation tables of all configured splits.
        config: Resolved configuration with `train_splits` and the tables of
            `build_train_transform`.
        frame_step: Frame step of every training clip.
        is_clean: Whether the frames get the transform of the clean epochs.

    Returns:
        A `VisDroneDataset` of the train clips, or with val in `train_splits` a
        `ConcatDataset` of the train and val clips; both splits get the same transform.
    """
    transform = build_train_transform(config, is_clean)
    datasets = [
        VisDroneDataset(tables, split, frame_step, transform) for split in config["train_splits"]
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_train_loader(
    dataset: Dataset, config: dict, device: torch.device, generator: torch.Generator
) -> DataLoader:
    """Build the shuffling DataLoader of the training frames.

    Args:
        dataset: Training frames.
        config: Resolved configuration with `batch_size` and `num_workers`.
        device: Device to train on; CUDA pins host memory.
        generator: Seeded generator of the shuffling order.

    Returns:
        The DataLoader with `collate_detection_batch` and persistent workers.
    """
    num_workers = config["num_workers"]
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def fit_torchvision_detector(
    model: torch.nn.Module,
    config: dict,
    train_dataset: Dataset,
    monitor_tables: AnnotationTables | None,
    checkpoint_dir: Path,
    device: torch.device,
    wandb_run: wandb.Run,
    clean_dataset: Dataset | None = None,
) -> tuple[list[dict], int]:
    """Train a detector of the shared loop, monitor it after every epoch, and save checkpoints.

    The global RNG and the shuffling generator are seeded with `seed`. With
    `freeze_batchnorm`, BatchNorm parameters are left out of the optimizer and BatchNorm
    layers keep their pretrained running statistics. Faster R-CNN gets the box regression
    loss of `[roi_heads]` (`set_box_regression_loss`), RetinaNet the focal loss of
    `[focal_loss]` (`set_focal_loss`). The family's optimizer gets the LR
    schedule of `build_lr_scheduler` over all optimizer steps of the run (`epochs` times the
    batches per epoch divided by `grad_accum`, rounded up); float16 training scales the
    loss. Each epoch trains
    on shuffled batches of `batch_size` frames (`train_one_epoch`), then predicts the
    monitoring frames one at a time in eval mode and scores them with the shared evaluator.
    `last.pt` is written after every epoch and `best.pt` whenever the monitoring mAP beats
    all earlier epochs (ties keep the earlier epoch); both hold `model` (the state dict) and
    `epoch`. Without monitoring frames (val trained on), `best.pt` is written after every
    epoch, so it holds the last one. The epoch record and a few frames with GT and predicted
    boxes go to W&B.
    With `[augment]` `train_scales`, every batch of an epoch gets an input size from
    `draw_input_sizes`, seeded with `seed` and the epoch. The last `clean_epochs` epochs
    train at the configured input size and on `clean_dataset` when given, the official
    RT-DETRv2 recipe's switch-off of augmentation before the end of training. With `[ema]`
    `enabled`, a copy of the model keeps an exponential moving average of the weights
    (`update_ema_weights`), and the monitoring and both checkpoints use that copy.

    Args:
        model: torchvision detector with background label 0, or wrapped RT-DETR.
        config: Resolved configuration with `run.family`, `epochs`, `batch_size`,
            `num_workers`, `grad_accum`, `amp`, `amp_dtype`, `grad_clip_norm`, `seed`,
            `freeze_batchnorm`, `schedule`, `warmup_iters`, the fields of the family's
            optimizer builder, the `[roi_heads]` table for Faster R-CNN, the `[focal_loss]`
            table for RetinaNet, and the `[augment]`, `[ema]`, and `[logging]` tables.
        train_dataset: Training frames.
        monitor_tables: Annotation tables whose `images` are the monitoring val frames, or
            None when val is trained on and no frames are held out.
        checkpoint_dir: Existing directory for `best.pt` and `last.pt`.
        device: Device to train on.
        wandb_run: W&B run for curves and sample frames.
        clean_dataset: Training frames without the scheduled augmentation, for the clean
            epochs, or None to train every epoch on `train_dataset`.

    Returns:
        One record per epoch (`epoch`, mean `train/` losses, `train/lr`, `val/` monitoring
        metrics unless `monitor_tables` is None, `epoch_seconds`, and `gpu_memory_gb`, the
        peak allocated memory during training or None on CPU) and the 1-based epoch saved as
        `best.pt`.
    """
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = build_train_loader(train_dataset, config, device, generator)
    clean_loader = (
        None
        if clean_dataset is None
        else build_train_loader(clean_dataset, config, device, generator)
    )
    train_scales = config["augment"]["train_scales"]
    first_clean_epoch = config["epochs"] - config["augment"]["clean_epochs"] + 1
    if isinstance(model, FasterRCNN):
        set_box_regression_loss(model, config["roi_heads"])
    if isinstance(model, RetinaNet):
        set_focal_loss(model, config["focal_loss"])
    model.to(device)
    if config["freeze_batchnorm"]:
        freeze_batchnorm(model)
    ema_model = copy.deepcopy(model).requires_grad_(False) if config["ema"]["enabled"] else None
    evaluated_model = model if ema_model is None else ema_model
    family = config["run"]["family"]
    optimizer = OPTIMIZER_BUILDERS[family](model, config)
    total_steps = config["epochs"] * math.ceil(len(loader) / config["grad_accum"])
    scheduler = build_lr_scheduler(
        optimizer, config["schedule"], config["warmup_iters"], total_steps
    )
    is_cuda = device.type == "cuda"
    autocast_dtype = select_autocast_dtype(config, device)
    logger.info("Training precision %s", describe_precision(autocast_dtype))
    scaler = torch.amp.GradScaler(device.type, enabled=autocast_dtype == torch.float16)
    logging_config = config["logging"]
    for metric, step_metric in (("iter/*", "iteration"), ("train/*", "epoch"), ("val/*", "epoch")):
        wandb_run.define_metric(metric, step_metric=step_metric)
    history, best_map, best_epoch, iteration = [], -1.0, 0, 0
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        is_clean_epoch = epoch >= first_clean_epoch
        epoch_loader = clean_loader if is_clean_epoch and clean_loader is not None else loader
        input_sizes = None
        if train_scales and not is_clean_epoch:
            input_sizes = draw_input_sizes(
                train_scales,
                config["min_size"],
                config["max_size"],
                len(epoch_loader),
                np.random.default_rng([config["seed"], epoch]),
            )
        losses, iteration = train_one_epoch(
            model,
            epoch_loader,
            optimizer,
            scheduler,
            scaler,
            config,
            device,
            wandb_run,
            iteration,
            input_sizes,
            ema_model,
        )
        gpu_memory_gb = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB if is_cuda else None
        record = {
            "epoch": epoch,
            **{f"train/{name}": value for name, value in losses.items()},
            "train/lr": optimizer.param_groups[0]["lr"],
        }
        checkpoint = {MODEL_STATE_KEY: evaluated_model.state_dict(), "epoch": epoch}
        if monitor_tables is None:
            record |= {
                "epoch_seconds": time.perf_counter() - started,
                "gpu_memory_gb": gpu_memory_gb,
            }
            history.append(record)
            best_epoch = epoch
            for checkpoint_name in (LAST_CHECKPOINT, BEST_CHECKPOINT):
                torch.save(checkpoint, checkpoint_dir / checkpoint_name)
            wandb_run.log(record)
            logger.info(
                "Epoch %d/%d: train loss %.4f, no held-out val frames to monitor",
                epoch,
                config["epochs"],
                losses["loss"],
            )
            continue
        predictions, _ = detect_frames(
            evaluated_model,
            monitor_tables.images,
            device,
            warmup_images=0,
            has_background_label=family in TORCHVISION_FAMILIES,
        )
        metrics = evaluate_detections(monitor_tables, "val", predictions)
        val_map = metrics["map"]
        if val_map is None:
            raise ValueError("The monitoring frames have no labelled objects")
        record |= {
            **{f"val/{metric}": metrics[metric] for metric in MONITORED_METRICS},
            "epoch_seconds": time.perf_counter() - started,
            "gpu_memory_gb": gpu_memory_gb,
        }
        history.append(record)
        torch.save(checkpoint, checkpoint_dir / LAST_CHECKPOINT)
        if val_map > best_map:
            best_map, best_epoch = val_map, epoch
            torch.save(checkpoint, checkpoint_dir / BEST_CHECKPOINT)
        sample_images = select_sample_frames(monitor_tables.images, logging_config["sample_frames"])
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
    prediction from `best.pt` with one warmup pass. With val in `train_splits`, the epoch
    also trains on those two val frames and skips the monitoring. The single epoch uses the
    configured augmentation, with `train_scales` scaled to the smoke input size and no clean
    epochs.
    Checkpoints go to a temporary directory that is removed afterwards.

    Args:
        config: Resolved configuration.
        tables: Annotation tables of all configured splits.
        wandb_run: Disabled W&B run.
    """
    smoke_config = config | SMOKE_OVERRIDES
    scale = SMOKE_OVERRIDES["min_size"] / config["min_size"]
    smoke_scales = [round(size * scale) for size in config["augment"]["train_scales"]]
    smoke_config |= {
        "inference": config["inference"] | {"warmup_images": SMOKE_WARMUP_IMAGES},
        "logging": config["logging"] | SMOKE_LOGGING,
        "augment": config["augment"] | {"train_scales": smoke_scales, "clean_epochs": 0},
    }
    model = build_model(smoke_config)
    smoke_images = (
        select_smoke_frames(tables.images, DATA_CONFIG_PATH)
        .sort("split", "frame_id")
        .group_by("split", maintain_order=True)
        .head(SMOKE_FRAMES_PER_SPLIT)
    )
    val_frames = smoke_images.filter(pl.col("split") == "val")
    train_dataset = build_train_dataset(
        tables._replace(images=smoke_images), smoke_config, frame_step=1, is_clean=False
    )
    is_val_trained = "val" in config["train_splits"]
    with tempfile.TemporaryDirectory() as work_dir:
        checkpoint_dir = Path(work_dir)
        fit_torchvision_detector(
            model,
            smoke_config,
            train_dataset,
            None if is_val_trained else tables._replace(images=val_frames),
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
    """Train a detector of the shared loop on the adopted split and record the run.

    Trains on the `train_frames` step of the `train_splits` clips (`build_train_dataset`) and
    monitors on the `val_monitoring` step of the val clips (`fit_torchvision_detector`),
    writing `best.pt` and `last.pt` to `run_dir` and the `training` section of
    `metrics.json`. With val in `train_splits`, no frames are held out: monitoring is skipped,
    `best.pt` is the last epoch, and `val_frame_step` is None. Training frames get the
    transform of `build_train_transform`; with flips, photometric distortion, or
    low-resolution frames and `clean_epochs`, the clean epochs use a second dataset with the
    clean transform. A smoke run
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
    augment = config["augment"]
    train_dataset = build_train_dataset(tables, config, train_step, is_clean=False)
    has_clean_dataset = augment["clean_epochs"] > 0 and (
        augment["hflip"] or augment["photometric"] or augment["low_res_prob"] > 0
    )
    clean_dataset = (
        build_train_dataset(tables, config, train_step, is_clean=True)
        if has_clean_dataset
        else None
    )
    is_val_trained = "val" in config["train_splits"]
    val_images = tables.images.filter(pl.col("split") == "val")
    monitor_tables = (
        None if is_val_trained else tables._replace(images=select_frames(val_images, val_step))
    )
    device = select_device()
    started = time.perf_counter()
    history, best_epoch = fit_torchvision_detector(
        model, config, train_dataset, monitor_tables, run_dir, device, wandb_run, clean_dataset
    )
    gpu_hours = (time.perf_counter() - started) / 3600
    training = describe_run(config, REPO_ROOT) | {
        "frame_step": train_step,
        "val_frame_step": None if is_val_trained else val_step,
        "epochs": len(history),
        "batch_size": config["batch_size"],
        "grad_accum": config["grad_accum"],
        "effective_batch_size": config["batch_size"] * config["grad_accum"],
        "precision": describe_precision(select_autocast_dtype(config, device)),
        "gpu_hours": gpu_hours,
        "wandb_url": wandb_run.url,
        "history": history,
        "best_epoch": best_epoch,
    }
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training}, indent=2) + "\n")
    logger.info("Best epoch %d of %d; wrote %s", best_epoch, len(history), run_dir / METRICS_FILE)
