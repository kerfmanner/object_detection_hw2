from pathlib import Path

import torch
import wandb


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
        config: Resolved configuration from `load_experiment_config`.
        data_root: Dataset root with the provided splits.
        run_dir: Run directory, or None for a smoke run.
        wandb_run: W&B run for curves and sample predictions.
    """
    raise NotImplementedError("The training loop is added together with the first model family")
