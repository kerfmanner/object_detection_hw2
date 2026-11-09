import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl
import torch
import wandb
from ultralytics import YOLO, settings
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import unwrap_model

from src.data.annotations import MAIN_CLIP_SPLIT, load_annotations, load_clip_split
from src.data.dataset import load_frame_steps, select_smoke_frames
from src.data.yolo_export import write_yolo_dataset
from src.evaluate import DATA_CONFIG_PATH, METRICS_FILE
from src.inference.frames import predict_frames
from src.models.yolo import set_assigner_topk
from src.submission.writer import format_submission
from src.training.experiment import REPO_ROOT, TRAIN_FRAME_STEPS, describe_run

logger = logging.getLogger(__name__)

DATASET_DIR = "yolo_data"
ULTRALYTICS_DIR = "ultralytics"
CHECKPOINT_FILES = ("best.pt", "last.pt")
AUTOBATCH = -1
SMOKE_TRAIN_OPTIONS = {"epochs": 1, "device": "cpu", "workers": 0, "cache": False, "plots": False}
SMOKE_IMGSZ = 320
SMOKE_WARMUP_IMAGES = 1
SMOKE_FRAMES_PER_SPLIT = 2
SAMPLE_PLOTS_PATTERN = "val_batch*_*.jpg"


def check_training_budget(config: dict) -> None:
    """Check the clip split, training frames, epochs, and batch size of a full YOLO run.

    Args:
        config: Resolved configuration with `train_frames`, `clip_split`, `epochs`, and
            `batch_size`.
    """
    problems = []
    if config["clip_split"] != MAIN_CLIP_SPLIT:
        problems.append(f"clip_split must be {MAIN_CLIP_SPLIT}; folds need the shared loop")
    if config["train_frames"] not in TRAIN_FRAME_STEPS:
        problems.append(f"train_frames must be one of {TRAIN_FRAME_STEPS}")
    if type(config["epochs"]) is not int or config["epochs"] < 1:
        problems.append("epochs must be an integer of at least 1")
    batch_size = config["batch_size"]
    if type(batch_size) is not int or (batch_size < 1 and batch_size != AUTOBATCH):
        problems.append(f"batch_size must be an integer of at least 1, or {AUTOBATCH}")
    if problems:
        raise ValueError(f"Invalid YOLO config {config['run']['config']}: {'; '.join(problems)}")


def compute_effective_batch_size(trainer: BaseTrainer) -> int:
    """Compute the images per optimizer step, including Ultralytics gradient accumulation.

    Ultralytics accumulates gradients over `round(nbs / batch)` batches after warmup:
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py

    Args:
        trainer: Trainer after training, with the batch size it actually used.

    Returns:
        Batch size times the number of accumulated batches.
    """
    return trainer.batch_size * max(round(trainer.args.nbs / trainer.batch_size), 1)


def fit_yolo(
    model: YOLO,
    config: dict,
    data_yaml: Path,
    save_dir: Path,
    wandb_run: wandb.Run,
    overrides: dict,
) -> tuple[BaseTrainer, int]:
    """Train a YOLO model with Ultralytics, writing all its outputs into `save_dir`.

    Training hyperparameters not set here are Ultralytics defaults; the TAL topk of
    `[assigner]` is set on the training and EMA models when training starts. Per-epoch
    validation on the `val` list of `data_yaml` selects `best.pt` by mAP@[.5:.95] and uses
    the NMS options of `[inference]`. Each epoch's losses, metrics, and learning rates go to
    W&B through our Ultralytics callbacks (https://docs.ultralytics.com/usage/callbacks/);
    the built-in Ultralytics W&B integration is switched off in the Ultralytics settings
    (https://docs.ultralytics.com/quickstart/#modifying-ultralytics-settings), so each run
    is logged once.

    Args:
        model: YOLO model from `build_model`.
        config: Resolved configuration with `imgsz`, `epochs`, `batch_size`, `optimizer`, `lr`,
            `momentum`, `warmup_bias_lr`, `close_mosaic`, `scale`, `amp`, `seed`, `[assigner]`
            `topk`, and `[inference]` `iou`, `max_det`, and `agnostic_nms`.
        data_yaml: Path of the `data.yaml` from `write_yolo_dataset`.
        save_dir: New directory for the Ultralytics outputs.
        wandb_run: W&B run for the per-epoch curves.
        overrides: Ultralytics train arguments that replace the configured ones.

    Returns:
        The trainer after training and the 1-based epoch saved as `best.pt`.
    """
    best_epochs = []
    finished_trainers: list[BaseTrainer] = []

    def record_best_epoch(trainer: BaseTrainer) -> None:
        if trainer.best_fitness == trainer.fitness:
            best_epochs.append(trainer.epoch + 1)

    def set_trainer_topk(trainer: BaseTrainer) -> None:
        if not isinstance(trainer.model, torch.nn.Module) or trainer.ema is None:
            raise RuntimeError("Ultralytics started training without a model and its EMA")
        for detector in (unwrap_model(trainer.model), trainer.ema.ema):
            set_assigner_topk(detector, config["assigner"]["topk"])

    def log_epoch(trainer: BaseTrainer) -> None:
        wandb_run.log(
            {
                **trainer.label_loss_items(trainer.tloss, prefix="train"),
                **(trainer.metrics or {}),
                **trainer.lr,
                "epoch": trainer.epoch + 1,
            }
        )

    settings.update({"wandb": False})
    model.add_callback("on_model_save", record_best_epoch)
    model.add_callback("on_fit_epoch_end", log_epoch)
    model.add_callback("on_train_end", finished_trainers.append)
    model.add_callback("on_train_start", set_trainer_topk)
    inference = config["inference"]
    options = {
        "data": str(data_yaml),
        "imgsz": config["imgsz"],
        "epochs": config["epochs"],
        "batch": config["batch_size"],
        "optimizer": config["optimizer"],
        "lr0": config["lr"],
        "momentum": config["momentum"],
        "warmup_bias_lr": config["warmup_bias_lr"],
        "close_mosaic": config["close_mosaic"],
        "scale": config["scale"],
        "amp": config["amp"],
        "seed": config["seed"],
        "iou": inference["iou"],
        "max_det": inference["max_det"],
        "agnostic_nms": inference["agnostic_nms"],
        "device": 0 if torch.cuda.is_available() else "cpu",
        "project": str(save_dir.parent.resolve()),
        "name": save_dir.name,
        "exist_ok": False,
    }
    model.train(**(options | overrides))
    return finished_trainers[-1], best_epochs[-1]


def run_yolo_smoke(model: YOLO, config: dict, images: pl.DataFrame, wandb_run: wandb.Run) -> None:
    """Train one CPU step on two smoke frames and predict two smoke val frames.

    The first two frames of the smoke train and val clips are exported at imgsz 320; the
    batch holds both train frames, so the epoch is a single step, followed by one
    validation pass. Prediction uses one warmup pass. All outputs go to a temporary
    directory that is removed afterwards.

    Args:
        model: YOLO model from `build_model`.
        config: Resolved configuration.
        images: Image table of all configured splits.
        wandb_run: Disabled W&B run.
    """
    smoke_images = (
        select_smoke_frames(images, DATA_CONFIG_PATH)
        .sort("split", "frame_id")
        .group_by("split", maintain_order=True)
        .head(SMOKE_FRAMES_PER_SPLIT)
    )
    val_frames = smoke_images.filter(pl.col("split") == "val")
    smoke_config = config | {
        "imgsz": SMOKE_IMGSZ,
        "inference": config["inference"] | {"warmup_images": SMOKE_WARMUP_IMAGES},
    }
    with tempfile.TemporaryDirectory() as work_dir:
        data_yaml = write_yolo_dataset(
            smoke_images, {"train": 1, "val": 1}, Path(work_dir) / DATASET_DIR
        )
        overrides = SMOKE_TRAIN_OPTIONS | {"batch": SMOKE_FRAMES_PER_SPLIT}
        trainer, _ = fit_yolo(
            model, smoke_config, data_yaml, Path(work_dir) / ULTRALYTICS_DIR, wandb_run, overrides
        )
        predictions, ms_per_image = predict_frames(model, trainer.best, val_frames, smoke_config)
    submission = format_submission(predictions, val_frames)
    logger.info(
        "Smoke run predicted %d boxes on %d frames (%.0f ms/image on CPU)",
        predictions.height,
        submission.height,
        ms_per_image,
    )


def train_yolo(
    model: YOLO, config: dict, data_root: Path, run_dir: Path | None, wandb_run: wandb.Run
) -> None:
    """Train a YOLO model on the adopted split and record the run in its directory.

    The train frames (`train_frames` step) and the monitoring val frames (`val_monitoring`
    step) are exported to `<run_dir>/yolo_data`, Ultralytics writes its outputs to
    `<run_dir>/ultralytics`, and its `best.pt` and `last.pt` are copied to `run_dir`.
    `metrics.json` gets the `training` section with the per-epoch Ultralytics history.

    Args:
        model: YOLO model from `build_model`.
        config: Resolved configuration from `load_experiment_config`.
        data_root: Dataset root with the provided splits.
        run_dir: Run directory, or None for a smoke run that writes nothing.
        wandb_run: W&B run for curves and sample validation batches.
    """
    images = load_annotations(data_root, load_clip_split(DATA_CONFIG_PATH, MAIN_CLIP_SPLIT)).images
    if run_dir is None:
        run_yolo_smoke(model, config, images, wandb_run)
        return
    check_training_budget(config)
    frame_steps = load_frame_steps(DATA_CONFIG_PATH)
    train_step = frame_steps[config["train_frames"]]
    split_steps = {"train": train_step, "val": frame_steps["val_monitoring"]}
    data_yaml = write_yolo_dataset(images, split_steps, run_dir / DATASET_DIR)
    started = time.perf_counter()
    trainer, best_epoch = fit_yolo(
        model, config, data_yaml, run_dir / ULTRALYTICS_DIR, wandb_run, {}
    )
    gpu_hours = (time.perf_counter() - started) / 3600
    for checkpoint in CHECKPOINT_FILES:
        shutil.copy2(trainer.wdir / checkpoint, run_dir / checkpoint)
    sample_plots = sorted(trainer.save_dir.glob(SAMPLE_PLOTS_PATTERN))
    wandb_run.log(
        {"val_samples": [wandb.Image(str(path), caption=path.stem) for path in sample_plots]}
    )
    history = pl.read_csv(trainer.csv).to_dicts()
    training = describe_run(config, REPO_ROOT) | {
        "frame_step": train_step,
        "val_frame_step": frame_steps["val_monitoring"],
        "epochs": len(history),
        "batch_size": trainer.batch_size,
        "effective_batch_size": compute_effective_batch_size(trainer),
        "gpu_hours": gpu_hours,
        "wandb_url": wandb_run.url,
        "history": history,
        "best_epoch": best_epoch,
    }
    (run_dir / METRICS_FILE).write_text(json.dumps({"training": training}, indent=2) + "\n")
    logger.info("Best epoch %d of %d; wrote %s", best_epoch, len(history), run_dir / METRICS_FILE)
