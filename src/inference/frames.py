from collections.abc import Callable
from pathlib import Path

import polars as pl
import torch

from src.inference.torchvision_detectors import predict_torchvision_frames
from src.inference.yolo import predict_yolo_frames

FramePredictor = Callable[..., tuple[pl.DataFrame, float]]

FAMILY_PREDICTORS: dict[str, FramePredictor] = {
    "faster_rcnn": predict_torchvision_frames,
    "retinanet": predict_torchvision_frames,
    "rtdetr": predict_torchvision_frames,
    "yolo": predict_yolo_frames,
}


def predict_frames(
    model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
) -> tuple[pl.DataFrame, float]:
    """Predict boxes on every frame of an image table with a trained checkpoint.

    Args:
        model: Detector from `build_model`.
        checkpoint_path: Path of the run's `best.pt`.
        images: Image table with `image_id`, `width`, `height`, and absolute `file_path`.
        config: Resolved run configuration; `run.family` selects the predictor and the
            `[inference]` table sets the options.

    Returns:
        Predictions with `image_id`, 0-based `class_id`, `score`, and pixel `x`, `y`, `w`,
        `h` (top-left corner), at most 500 per frame, and the mean inference time in ms per
        image.
    """
    family = config["run"]["family"]
    if family not in FAMILY_PREDICTORS:
        raise ValueError(
            f"No predictor registered for family {family!r}; "
            f"registered families: {sorted(FAMILY_PREDICTORS)}"
        )
    return FAMILY_PREDICTORS[family](model, checkpoint_path, images, config)
