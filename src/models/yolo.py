from ultralytics import YOLO

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
