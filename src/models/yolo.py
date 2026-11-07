import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

SUPPORTED_INIT = "coco"


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
