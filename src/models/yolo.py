import logging
import re
from collections.abc import Sequence

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

logger = logging.getLogger(__name__)

SUPPORTED_INIT = "coco"
POINT_KEY_ROW = 1 << 32
FIRST_P2_SHIFTED_LAYER = 17
P2_LAYER_SHIFT = 6
DETECT_LEVEL_PATTERN = re.compile(r"^(cv2|cv3|one2one_cv2|one2one_cv3)\.(\d+)\.")


def build_yolo(config: dict) -> YOLO:
    """Build an Ultralytics YOLO detector from its COCO-pretrained checkpoint.

    Training replaces the 80-class COCO head with one for the dataset classes and loads all
    other weights from the checkpoint: https://docs.ultralytics.com/modes/train/
    With `p2_head`, the model is built from the Ultralytics `<arch>-p2.yaml` architecture
    (an extra stride-4 detection level, e.g. `yolo26-p2.yaml` at the scale of `arch`) and
    gets the checkpoint weights through `load_p2_weights`, which leaves only the new P2
    layers randomly initialised: https://docs.ultralytics.com/models/yolo26/

    Args:
        config: Resolved configuration with `arch` (e.g. `yolo11s`), `init` (`coco`), and
            `p2_head`.

    Returns:
        The YOLO model with the weights `<arch>.pt`, downloaded by Ultralytics if missing.
    """
    if config["init"] != SUPPORTED_INIT:
        raise ValueError(f"YOLO supports init {SUPPORTED_INIT!r} only, got {config['init']!r}")
    if config["p2_head"]:
        model = YOLO(f"{config['arch']}-p2.yaml")
        source = YOLO(f"{config['arch']}.pt").model
        if not isinstance(model.model, DetectionModel) or not isinstance(source, DetectionModel):
            raise TypeError(f"{config['arch']} is not an Ultralytics detection model")
        transferred = load_p2_weights(model.model, source)
        logger.info(
            "Loaded %d of %d %s.pt tensors into %s-p2",
            transferred,
            len(source.state_dict()),
            config["arch"],
            config["arch"],
        )
        model.load(model.model)
        return model
    return YOLO(f"{config['arch']}.pt")


def load_p2_weights(detector: DetectionModel, source: DetectionModel) -> int:
    """Load the weights of a P3 to P5 YOLO26 detector into its P2 to P5 variant.

    `yolo26-p2.yaml` inserts the P2 layers 17 to 22 after the top-down path, so layers 0 to
    16 keep their index, layers 17 to 23 of `yolo26.yaml` (bottom-up path and `Detect`)
    become layers 23 to 29, and the source detection levels P3, P4, P5 (0, 1, 2) become
    levels 1, 2, 3 of the new `Detect`
    (https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26).
    A tensor is loaded only where the mapped name exists with the same shape.

    Args:
        detector: Detector built from `<arch>-p2.yaml`.
        source: Pretrained detector of the same scale built from `<arch>.yaml`.

    Returns:
        Number of source tensors loaded.
    """
    target = detector.state_dict()
    detect_index = max(int(key.split(".")[1]) for key in target)
    mapped = {}
    for key, value in source.state_dict().items():
        _, layer, name = key.split(".", 2)
        index = int(layer)
        if index >= FIRST_P2_SHIFTED_LAYER:
            index += P2_LAYER_SHIFT
        if index == detect_index:
            name = DETECT_LEVEL_PATTERN.sub(lambda m: f"{m[1]}.{int(m[2]) + 1}.", name)
        target_key = f"model.{index}.{name}"
        if target_key in target and target[target_key].shape == value.shape:
            mapped[target_key] = value
    detector.load_state_dict(mapped, strict=False)
    return len(mapped)


def set_assigner_topk(detector: torch.nn.Module, topk: int) -> None:
    """Give a YOLO detection model a detection loss with the configured TAL topk.

    Ultralytics creates `v8DetectionLoss` lazily with `tal_topk` 10 and has no train argument
    for it (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py);
    the task-aligned assigner keeps the `topk` grid points inside each GT box with the highest
    alignment metric as positives (TOOD, https://arxiv.org/abs/2108.07755). End-to-end models
    (YOLO26) train with `E2ELoss`, whose one-to-many branch is a `v8DetectionLoss` with
    `tal_topk` 10; only that branch gets the configured topk, and the one-to-one branch keeps
    its own assignment. The loss stays for the whole run: Ultralytics removes it only from
    saved checkpoint copies.

    Args:
        detector: Ultralytics `DetectionModel` on its training device, with the train
            arguments in `args`.
        topk: Positive grid points per GT box, at least 1.
    """
    if type(topk) is not int or topk < 1:
        raise ValueError(f"TAL topk must be an integer of at least 1, got {topk!r}")
    if not isinstance(detector, DetectionModel):
        raise TypeError(f"Expected an Ultralytics DetectionModel, got {type(detector).__name__}")
    criterion = detector.init_criterion()
    if type(criterion) is E2ELoss:
        criterion.one2many = v8DetectionLoss(detector, tal_topk=topk)
        detector.criterion = criterion
    elif type(criterion) is v8DetectionLoss:
        detector.criterion = v8DetectionLoss(detector, tal_topk=topk)
    else:
        raise TypeError(
            f"{type(detector).__name__} trains with neither v8DetectionLoss nor E2ELoss"
        )


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
