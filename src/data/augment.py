from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from torchvision.transforms import v2

FLIP_PROBABILITY = 0.5
PHOTOMETRIC_PROBABILITY = 0.5
FrameTransform = Callable[
    [torch.Tensor, dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, torch.Tensor]]
]


def mask_ignored_regions(
    image: torch.Tensor, ignored_boxes: torch.Tensor, object_boxes: torch.Tensor, fill: list[int]
) -> torch.Tensor:
    """Fill the ignored regions of a frame with one colour, except inside labelled boxes.

    Ignored regions are rounded to whole pixels; labelled boxes are widened to whole pixels,
    so every pixel a labelled box overlaps keeps its value, also inside ignored regions.

    Args:
        image: uint8 tensor (3, height, width).
        ignored_boxes: Tensor (k, 4) with `x1, y1, x2, y2` in px of the ignored regions.
        object_boxes: Tensor (n, 4) with `x1, y1, x2, y2` in px of the labelled objects.
        fill: RGB fill colour, three integers from 0 to 255.

    Returns:
        A new uint8 tensor of the same shape.
    """
    height, width = image.shape[-2:]
    mask = torch.zeros((height, width), dtype=torch.bool)
    for x1, y1, x2, y2 in ignored_boxes.round().int().tolist():
        mask[max(y1, 0) : y2, max(x1, 0) : x2] = True
    for x1, y1, x2, y2 in object_boxes.tolist():
        mask[max(int(y1), 0) : int(np.ceil(y2)), max(int(x1), 0) : int(np.ceil(x2))] = False
    masked = image.clone()
    masked[:, mask] = torch.tensor(fill, dtype=image.dtype)[:, None]
    return masked


def flip_frame(
    image: torch.Tensor, target: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flip a frame and the boxes of its target horizontally.

    Args:
        image: Tensor (3, height, width).
        target: Target with `boxes` and `ignored_boxes` (`x1, y1, x2, y2` in px); other keys
            are kept.

    Returns:
        The flipped frame and a new target with flipped `boxes` and `ignored_boxes`.
    """
    width = image.shape[-1]
    flipped = dict(target)
    for key in ("boxes", "ignored_boxes"):
        boxes = target[key]
        flipped[key] = torch.stack(
            [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
        )
    return image.flip(-1), flipped


def augment_training_frame(
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    mask_fill: list[int] | None,
    photometric: v2.RandomPhotometricDistort | None,
    is_flip_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mask, photometrically distort, and flip one training frame, each step optional.

    Args:
        image: uint8 tensor (3, height, width).
        target: `VisDroneDataset` target with `boxes`, `labels`, and `ignored_boxes`.
        mask_fill: RGB colour for `mask_ignored_regions`, or None to keep ignored regions.
        photometric: Photometric distortion of the image, or None.
        is_flip_enabled: Whether to flip the frame with probability 0.5.

    Returns:
        The augmented uint8 frame and its target.
    """
    if mask_fill is not None:
        image = mask_ignored_regions(image, target["ignored_boxes"], target["boxes"], mask_fill)
    if photometric is not None:
        image = photometric(image)
    if is_flip_enabled and torch.rand(()) < FLIP_PROBABILITY:
        image, target = flip_frame(image, target)
    return image, target


def build_train_transform(config: dict, is_clean: bool) -> FrameTransform | None:
    """Build the training frame transform of the `[augment]` and `[ignored_regions]` tables.

    The flip and the photometric distortion follow the official RT-DETRv2 training recipe
    (https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetrv2/include/dataloader.yml):
    `RandomHorizontalFlip` with p 0.5 and torchvision's `RandomPhotometricDistort` with p
    0.5 and its default ranges. Clean epochs, the recipe's last epochs without augmentation,
    keep only the masking of ignored regions, which is a separate data strategy.

    Args:
        config: Resolved configuration with the `[augment]` (`hflip`, `photometric`) and
            `[ignored_regions]` (`mask`, `fill`) tables.
        is_clean: Whether the transform is for the clean epochs.

    Returns:
        A transform of a uint8 frame and its target, or None when it would change nothing.
    """
    augment, ignored = config["augment"], config["ignored_regions"]
    mask_fill = ignored["fill"] if ignored["mask"] else None
    if mask_fill is not None and (
        len(mask_fill) != 3 or not all(0 <= value <= 255 for value in mask_fill)
    ):
        raise ValueError(f"ignored_regions.fill must be three values 0 to 255, got {mask_fill}")
    is_photometric = augment["photometric"] and not is_clean
    is_flip_enabled = augment["hflip"] and not is_clean
    if mask_fill is None and not is_photometric and not is_flip_enabled:
        return None
    return partial(
        augment_training_frame,
        mask_fill=mask_fill,
        photometric=(
            v2.RandomPhotometricDistort(p=PHOTOMETRIC_PROBABILITY) if is_photometric else None
        ),
        is_flip_enabled=is_flip_enabled,
    )


def draw_input_sizes(
    train_scales: list[int], min_size: int, max_size: int, count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Draw the input size of every batch of an epoch for multi-scale training.

    Each batch gets a shorter-side size drawn uniformly from `train_scales`, as the batch
    collate of the official RT-DETRv2 recipe draws one of its scales per batch; `max_size`
    is scaled by the same factor, so that the longer side does not cap the larger sizes.

    Args:
        train_scales: Candidate shorter-side sizes in px; repeated values are drawn more often.
        min_size: Shorter side of the run in px.
        max_size: Upper bound of the longer side of the run in px.
        count: Number of batches.
        rng: Random generator of the draw, e.g. seeded with the run seed and the epoch.

    Returns:
        One (min_size, max_size) pair in px per batch.
    """
    if not train_scales or min(train_scales) < 1:
        raise ValueError(f"train_scales must be positive sizes, got {train_scales}")
    sizes = rng.choice(np.array(train_scales), size=count)
    return [(int(size), round(max_size * int(size) / min_size)) for size in sizes]
