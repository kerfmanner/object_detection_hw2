from collections.abc import Sequence

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

SUPPORTED_INIT = "coco"
POINT_KEY_ROW = 1 << 32


def build_yolo(config: dict) -> YOLO:
    """Build an Ultralytics YOLO detector from its COCO-pretrained checkpoint.

    Training replaces the 80-class COCO head with one for the dataset classes and loads all
    other weights from the checkpoint: https://docs.ultralytics.com/modes/train/

    Args:
        config: Resolved configuration with `arch` (e.g. `yolo11s`) and `init` (`coco`).

    Returns:
        The YOLO model with the weights `<arch>.pt`, downloaded by Ultralytics if missing.
    """
    if config["init"] != SUPPORTED_INIT:
        raise ValueError(f"YOLO supports init {SUPPORTED_INIT!r} only, got {config['init']!r}")
    return YOLO(f"{config['arch']}.pt")


def set_assigner_topk(detector: torch.nn.Module, topk: int) -> None:
    """Give a YOLO detection model a detection loss with the configured TAL topk.

    Ultralytics creates `v8DetectionLoss` lazily with `tal_topk` 10 and has no train argument
    for it (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py);
    the task-aligned assigner keeps the `topk` grid points inside each GT box with the highest
    alignment metric as positives (TOOD, https://arxiv.org/abs/2108.07755). The loss stays for
    the whole run: Ultralytics removes it only from saved checkpoint copies.

    Args:
        detector: Ultralytics `DetectionModel` on its training device, with the train
            arguments in `args`.
        topk: Positive grid points per GT box, at least 1.
    """
    if type(topk) is not int or topk < 1:
        raise ValueError(f"TAL topk must be an integer of at least 1, got {topk!r}")
    if not isinstance(detector, DetectionModel):
        raise TypeError(f"Expected an Ultralytics DetectionModel, got {type(detector).__name__}")
    if type(detector.init_criterion()) is not v8DetectionLoss:
        raise TypeError(f"{type(detector).__name__} does not train with v8DetectionLoss")
    detector.criterion = v8DetectionLoss(detector, tal_topk=topk)


def count_assignment_candidates(
    boxes: np.ndarray, strides: Sequence[int], min_side: float
) -> tuple[np.ndarray, np.ndarray]:
    """Count the grid points that the task-aligned assigner may assign to each box of a frame.

    A grid point of stride s sits at ((i + 0.5) s, (j + 0.5) s) of the network input; it is a
    candidate of a box if it lies strictly inside it, after box sides below `min_side` are
    enlarged to `min_side` about the box center, as in Ultralytics `select_candidates_in_gts`
    (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py). A point
    inside several boxes goes to only one of them (the one with the highest predicted IoU),
    so shared points are contested in dense scenes. The grid is assumed to start at the
    frame corner, so letterbox padding is ignored.

    Args:
        boxes: Array of shape (n, 4) with `x1, y1, x2, y2` in px of the network input.
        strides: Strides of the detection levels in px, e.g. 8, 16, 32.
        min_side: Box sides below this are enlarged to it (Ultralytics `stride_val`, 16).

    Returns:
        Two arrays of shape (n,): candidate points of each box over all strides, and how
        many of them also lie inside another box.
    """
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half_sides = np.maximum(boxes[:, 2:] - boxes[:, :2], min_side) / 2
    low, high = centers - half_sides, centers + half_sides
    candidates = np.zeros(len(boxes), dtype=np.int64)
    shared = np.zeros(len(boxes), dtype=np.int64)
    for stride in strides:
        first = np.floor(low / stride - 0.5).astype(np.int64) + 1
        last = np.ceil(high / stride - 0.5).astype(np.int64) - 1
        counts = np.clip(last - first + 1, 0, None)
        candidates += counts[:, 0] * counts[:, 1]
        box_index = np.repeat(np.arange(len(boxes)), counts[:, 0] * counts[:, 1])
        point_keys = np.concatenate(
            [
                (
                    np.arange(first[i, 1], last[i, 1] + 1)[:, None] * POINT_KEY_ROW
                    + np.arange(first[i, 0], last[i, 0] + 1)[None, :]
                ).ravel()
                for i in range(len(boxes))
            ]
            or [np.zeros(0, dtype=np.int64)]
        )
        _, inverse, multiplicity = np.unique(point_keys, return_inverse=True, return_counts=True)
        is_contested = multiplicity[inverse] > 1
        shared += np.bincount(box_index, weights=is_contested, minlength=len(boxes)).astype(
            np.int64
        )
    return candidates, shared
