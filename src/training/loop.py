from collections.abc import Callable
from pathlib import Path

import torch
import wandb

from src.training.torchvision_loop import train_torchvision_detector
from src.training.yolo import train_yolo

FamilyTrainer = Callable[..., None]

FAMILY_TRAINERS: dict[str, FamilyTrainer] = {
    "faster_rcnn": train_torchvision_detector,
    "retinanet": train_torchvision_detector,
    "rtdetr": train_torchvision_detector,
    "yolo": train_yolo,
}


def train_detector(
    model: torch.nn.Module,
    config: dict,
    data_root: Path,
    run_dir: Path | None,
    wandb_run: wandb.Run,
) -> None:
    """Train a detector and write `best.pt`, `last.pt`, and `metrics.json` to its run.

    A smoke run (`run_dir` None) does one CPU training step and one evaluation step on the
    smoke subset and writes nothing.

    Args:
        model: Detector from `build_model`.
        config: Resolved configuration from `load_experiment_config`; `run.family` selects
            the family's training.
        data_root: Dataset root with the provided splits.
        run_dir: Run directory, or None for a smoke run.
        wandb_run: W&B run for curves and sample predictions.
    """
    family = config["run"]["family"]
    if family not in FAMILY_TRAINERS:
        raise ValueError(
            f"No training registered for family {family!r}; "
            f"registered families: {sorted(FAMILY_TRAINERS)}"
        )
    FAMILY_TRAINERS[family](model, config, data_root, run_dir, wandb_run)
