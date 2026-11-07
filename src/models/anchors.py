from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator

DEFAULT_ANCHOR_MODELS = {
    "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn_v2": retinanet_resnet50_fpn_v2,
}
STRIDE_PROBE_SIZE = 256


def build_anchor_generator(
    sizes: list[list[int]], aspect_ratios: list[list[float]], pretrained: AnchorGenerator
) -> AnchorGenerator:
    """Build a configured anchor generator to replace the default one of a pretrained detector.

    The number of anchors per location must equal the pretrained generator's, so that the
    pretrained head weights still match it.

    Args:
        sizes: Anchor sizes in px (square root of the area), one list per FPN level.
        aspect_ratios: Anchor height / width ratios, one list per FPN level.
        pretrained: Anchor generator the detector was built with.

    Returns:
        Anchor generator with the configured sizes and aspect ratios.
    """
    generator = AnchorGenerator(
        tuple(tuple(level_sizes) for level_sizes in sizes),
        tuple(tuple(level_ratios) for level_ratios in aspect_ratios),
    )
    if generator.num_anchors_per_location() != pretrained.num_anchors_per_location():
        raise ValueError(
            f"Anchors per location {generator.num_anchors_per_location()} differ from the "
            f"pretrained head's {pretrained.num_anchors_per_location()}"
        )
    return generator


def compute_shape_iou(shapes_a: np.ndarray, shapes_b: np.ndarray) -> np.ndarray:
    """Compute the IoU of box shapes placed at a common center.

    This is the box similarity used for anchor clustering in YOLOv2:
    https://arxiv.org/abs/1612.08242

    Args:
        shapes_a: Array of shape (n, 2) with widths and heights in px.
        shapes_b: Array of shape (m, 2) with widths and heights in px.

    Returns:
        Array of shape (n, m) with IoU values in [0, 1].
    """
    intersection = np.minimum(shapes_a[:, None, 0], shapes_b[None, :, 0]) * np.minimum(
        shapes_a[:, None, 1], shapes_b[None, :, 1]
    )
    union = (
        (shapes_a[:, 0] * shapes_a[:, 1])[:, None]
        + (shapes_b[:, 0] * shapes_b[:, 1])[None, :]
        - intersection
    )
    return intersection / union


def cluster_box_shapes(
    shapes: np.ndarray, k: int, seed: int, max_iterations: int
) -> tuple[np.ndarray, float]:
    """Cluster box shapes with k-means using 1 − IoU as the distance, as in YOLOv2.

    Centroids start at `k` distinct shapes drawn with the seed and move to the mean shape of
    their members until no assignment changes; an empty cluster keeps its centroid.
    YOLOv2: https://arxiv.org/abs/1612.08242

    Args:
        shapes: Array of shape (n, 2) with widths and heights in px.
        k: Number of clusters.
        seed: Seed of the initial centroid draw.
        max_iterations: Upper limit of assignment updates.

    Returns:
        Centroids of shape (k, 2) sorted by area, and the mean IoU of each shape with its
        nearest centroid.
    """
    rng = np.random.default_rng(seed)
    distinct_shapes = np.unique(shapes, axis=0)
    centroids = distinct_shapes[rng.choice(len(distinct_shapes), size=k, replace=False)]
    assignments = np.full(len(shapes), -1)
    for _ in range(max_iterations):
        new_assignments = compute_shape_iou(shapes, centroids).argmax(axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for cluster in range(k):
            members = shapes[assignments == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    centroids = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    mean_iou = float(compute_shape_iou(shapes, centroids).max(axis=1).mean())
    return centroids, mean_iou


def load_default_anchors(architecture: str) -> pl.DataFrame:
    """Read the default anchors of a torchvision detector from the installed library.

    The model is built without weights; strides come from running its backbone on a blank
    probe image.

    Args:
        architecture: Key of `DEFAULT_ANCHOR_MODELS`.

    Returns:
        One row per anchor with feature `level`, `stride` in px, corner offsets `x1`, `y1`,
        `x2`, `y2` from the anchor center in px, and `w`, `h` in px.
    """
    model = DEFAULT_ANCHOR_MODELS[architecture](weights=None, weights_backbone=None).eval()
    generator = model.rpn.anchor_generator if hasattr(model, "rpn") else model.anchor_generator
    with torch.no_grad():
        features = model.backbone(torch.zeros(1, 3, STRIDE_PROBE_SIZE, STRIDE_PROBE_SIZE))
    strides = [STRIDE_PROBE_SIZE // feature.shape[-1] for feature in features.values()]
    return tabulate_anchors(generator.sizes, generator.aspect_ratios, strides)


def tabulate_anchors(
    sizes: Sequence[Sequence[float]],
    aspect_ratios: Sequence[Sequence[float]],
    strides: Sequence[int],
) -> pl.DataFrame:
    """List the cell anchors that torchvision's `AnchorGenerator` builds for an anchor setting.

    Args:
        sizes: Anchor sizes in px (square root of the area), one sequence per FPN level.
        aspect_ratios: Anchor height / width ratios, one sequence per FPN level.
        strides: Stride in px of each FPN level.

    Returns:
        One row per anchor with feature `level`, `stride` in px, corner offsets `x1`, `y1`,
        `x2`, `y2` from the anchor center in px, and `w`, `h` in px.
    """
    generator = AnchorGenerator(
        tuple(tuple(level_sizes) for level_sizes in sizes),
        tuple(tuple(level_ratios) for level_ratios in aspect_ratios),
    )
    rows = [
        (level, stride, *corners)
        for level, (stride, cell_anchors) in enumerate(
            zip(strides, generator.cell_anchors, strict=True)
        )
        for corners in cell_anchors.tolist()
    ]
    return pl.DataFrame(
        rows, schema=["level", "stride", "x1", "y1", "x2", "y2"], orient="row"
    ).with_columns(
        (pl.col("x2") - pl.col("x1")).alias("w"), (pl.col("y2") - pl.col("y1")).alias("h")
    )


def compute_best_anchor_iou(
    boxes: np.ndarray, anchors: pl.DataFrame, is_grid_aligned: bool, chunk_size: int = 100_000
) -> np.ndarray:
    """Compute for every box the largest IoU with any anchor.

    Grid-aligned anchors sit at the centers `(i * stride, j * stride)` used by torchvision's
    `AnchorGenerator`; each box is compared with anchors at the four grid points around its
    center on every level. Otherwise anchors and boxes share a center, as in YOLOv2.

    Args:
        boxes: Array of shape (n, 4) with `x, y, w, h` in px of the resized frame.
        anchors: Anchor table from `load_default_anchors` or with at least `w` and `h`.
        is_grid_aligned: Whether to place anchors on their feature-map grid.
        chunk_size: Boxes processed at once, to bound memory.

    Returns:
        Array of shape (n,) with the best IoU of each box.
    """
    if not is_grid_aligned:
        anchor_shapes = anchors.select("w", "h").to_numpy().astype(np.float64)
        return np.concatenate(
            [
                compute_shape_iou(boxes[start : start + chunk_size, 2:], anchor_shapes).max(axis=1)
                for start in range(0, len(boxes), chunk_size)
            ]
        )
    best = np.zeros(len(boxes))
    for (stride,), level_anchors in anchors.group_by("stride"):
        offsets = level_anchors.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)
        for start in range(0, len(boxes), chunk_size):
            chunk = boxes[start : start + chunk_size]
            box_min, box_max = chunk[:, :2], chunk[:, :2] + chunk[:, 2:]
            grid_start = np.floor((box_min + box_max) / 2 / stride) * stride
            for dx in (0, stride):
                for dy in (0, stride):
                    center = grid_start + np.array([dx, dy])
                    anchor_min = center[:, None, :] + offsets[None, :, :2]
                    anchor_max = center[:, None, :] + offsets[None, :, 2:]
                    overlap = np.clip(
                        np.minimum(box_max[:, None, :], anchor_max)
                        - np.maximum(box_min[:, None, :], anchor_min),
                        0,
                        None,
                    )
                    intersection = overlap[..., 0] * overlap[..., 1]
                    union = (
                        (chunk[:, 2] * chunk[:, 3])[:, None]
                        + ((offsets[:, 2] - offsets[:, 0]) * (offsets[:, 3] - offsets[:, 1]))[None]
                        - intersection
                    )
                    np.maximum(
                        best[start : start + chunk_size],
                        (intersection / union).max(axis=1),
                        out=best[start : start + chunk_size],
                    )
    return best
