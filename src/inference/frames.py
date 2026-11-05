from pathlib import Path

import polars as pl
import torch


def predict_frames(
    model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
) -> tuple[pl.DataFrame, float]:
    """Predict boxes on every frame of an image table with a trained checkpoint.

    Args:
        model: Detector from `build_model`.
        checkpoint_path: Path of the run's `best.pt`.
        images: Image table with `image_id`, `width`, `height`, and absolute `file_path`.
        config: Resolved run configuration; its `[inference]` table sets the options.

    Returns:
        Predictions with `image_id`, 0-based `class_id`, `score`, and pixel `x`, `y`, `w`,
        `h` (top-left corner), at most 500 per frame, and the mean inference time in ms per
        image.
    """
    raise NotImplementedError("Inference is added together with the first model family")
