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
    """Drop detections of the background label 0 and shift the other labels by -1.

    Faster R-CNN never predicts label 0. RetinaNet scores every column of its sigmoid
    classifier, including column 0, which no target trains as positive, so it can predict
    label 0 with low scores.

    Args:
        detections: Per image, torchvision `boxes` (k, 4), `scores` (k,), and `labels` (k,)
            from 0 to 8.

    Returns:
        New detection dictionaries without label 0, with 0-based `labels` from 0 to 7.
    """
    shifted = []
    for frame_detections in detections:
        is_object = frame_detections["labels"] >= TORCHVISION_LABEL_OFFSET
        shifted.append(
            {
                "boxes": frame_detections["boxes"][is_object],
                "scores": frame_detections["scores"][is_object],
                "labels": frame_detections["labels"][is_object] - TORCHVISION_LABEL_OFFSET,
            }
        )
    return shifted
