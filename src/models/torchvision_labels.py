import torch

TORCHVISION_LABEL_OFFSET = 1


def shift_targets_to_torchvision(
    targets: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Shift 0-based target labels by +1 for a torchvision detector.

    torchvision detectors reserve label 0 for the background class:
    https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    Args:
        targets: Per image, `boxes` (n, 4) with `x1, y1, x2, y2` in px and 0-based `labels`
            (n,); other keys are dropped.

    Returns:
        New target dictionaries with the same `boxes` and labels from 1 to 8.
    """
    return [
        {"boxes": target["boxes"], "labels": target["labels"] + TORCHVISION_LABEL_OFFSET}
        for target in targets
    ]


def shift_detections_from_torchvision(
    detections: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Shift the labels of torchvision detections by -1 to our 0-based class ids.

    Args:
        detections: Per image, torchvision `boxes` (k, 4), `scores` (k,), and `labels` (k,)
            from 1 to 8.

    Returns:
        New detection dictionaries with the same `boxes` and `scores` and 0-based `labels`.
    """
    return [
        frame_detections | {"labels": frame_detections["labels"] - TORCHVISION_LABEL_OFFSET}
        for frame_detections in detections
    ]
