import numpy as np
import polars as pl

from src.data.annotations import BOX_COLUMNS


def convert_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert pixel boxes from top-left corner and size to corner coordinates.

    Args:
        boxes: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.

    Returns:
        Array of shape (n, 4) with `x1, y1, x2, y2` in px.
    """
    return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1)


def convert_xywh_to_yolo(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel boxes to YOLO boxes normalized by the frame size.

    Args:
        boxes: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
        width: Frame width in px.
        height: Frame height in px.

    Returns:
        Array of shape (n, 4) with `x_center, y_center, w, h` in [0, 1].
    """
    scale = np.array([width, height, width, height])
    return np.concatenate([boxes[:, :2] + boxes[:, 2:] / 2, boxes[:, 2:]], axis=1) / scale


def convert_yolo_to_xywh(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert YOLO boxes normalized by the frame size to pixel boxes.

    Args:
        boxes: Array of shape (n, 4) with `x_center, y_center, w, h` in [0, 1].
        width: Frame width in px.
        height: Frame height in px.

    Returns:
        Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
    """
    pixel_boxes = boxes * np.array([width, height, width, height])
    return np.concatenate([pixel_boxes[:, :2] - pixel_boxes[:, 2:] / 2, pixel_boxes[:, 2:]], axis=1)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute the IoU of every pair of boxes in two sets.

    Args:
        boxes_a: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
        boxes_b: Array of shape (m, 4) in the same format.

    Returns:
        Array of shape (n, m) with IoU values in [0, 1].
    """
    a_min, a_max = boxes_a[:, None, :2], boxes_a[:, None, :2] + boxes_a[:, None, 2:]
    b_min, b_max = boxes_b[None, :, :2], boxes_b[None, :, :2] + boxes_b[None, :, 2:]
    overlap = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None)
    intersection = overlap[..., 0] * overlap[..., 1]
    union = (
        (boxes_a[:, 2] * boxes_a[:, 3])[:, None]
        + (boxes_b[:, 2] * boxes_b[:, 3])[None, :]
        - intersection
    )
    return intersection / union


def compute_coverage_matrix(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Compute the fraction of each box covered by each region separately.

    Args:
        boxes: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
        regions: Array of shape (k, 4) in the same format.

    Returns:
        Array of shape (n, k) with intersection / box area in [0, 1].
    """
    box_min, box_max = boxes[:, None, :2], boxes[:, None, :2] + boxes[:, None, 2:]
    region_min, region_max = regions[None, :, :2], regions[None, :, :2] + regions[None, :, 2:]
    overlap = np.clip(np.minimum(box_max, region_max) - np.maximum(box_min, region_min), 0, None)
    return overlap[..., 0] * overlap[..., 1] / (boxes[:, 2] * boxes[:, 3])[:, None]


def compute_max_iou_with_others(boxes: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Compute for every box the largest IoU with any other box of the same set.

    Args:
        boxes: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
        groups: Optional array of shape (n,); only boxes with equal group values are compared.

    Returns:
        Array of shape (n,) with the maximum IoU, 0 for a box without partners.
    """
    iou = compute_iou_matrix(boxes, boxes)
    np.fill_diagonal(iou, 0.0)
    if groups is not None:
        iou[groups[:, None] != groups[None, :]] = 0.0
    return iou.max(axis=1, initial=0.0)


def add_max_iou_columns(objects: pl.DataFrame) -> pl.DataFrame:
    """Add the largest IoU of each object with another object of its frame.

    Args:
        objects: Object table with `split`, `image_id`, `class_id`, and pixel box columns.

    Returns:
        The object table with `max_iou_any_class` and `max_iou_same_class` columns.
    """
    frames = []
    for frame_objects in objects.partition_by("split", "image_id", maintain_order=True):
        boxes = frame_objects.select(BOX_COLUMNS).to_numpy().astype(np.float64)
        class_ids = frame_objects["class_id"].to_numpy()
        frames.append(
            frame_objects.with_columns(
                pl.Series("max_iou_any_class", compute_max_iou_with_others(boxes)),
                pl.Series("max_iou_same_class", compute_max_iou_with_others(boxes, class_ids)),
            )
        )
    return pl.concat(frames)


def compute_covered_fraction(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Compute the fraction of each box area covered by the union of a set of regions.

    The region edges split the plane into a grid of cells that are either fully inside or
    fully outside the union; each box sums its overlap with the covered cells.

    Args:
        boxes: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
        regions: Array of shape (k, 4) in the same format; regions may overlap each other.

    Returns:
        Array of shape (n,) with covered fractions in [0, 1].
    """
    if len(regions) == 0:
        return np.zeros(len(boxes))
    region_min, region_max = regions[:, :2], regions[:, :2] + regions[:, 2:]
    box_min, box_max = boxes[:, :2], boxes[:, :2] + boxes[:, 2:]
    cell_overlaps = []
    cell_inside = []
    for axis in (0, 1):
        edges = np.unique(np.concatenate([region_min[:, axis], region_max[:, axis]]))
        cell_start, cell_end = edges[:-1], edges[1:]
        cell_center = (cell_start + cell_end) / 2
        cell_inside.append(
            (region_min[:, axis, None] <= cell_center) & (cell_center < region_max[:, axis, None])
        )
        cell_overlaps.append(
            np.clip(
                np.minimum(box_max[:, axis, None], cell_end)
                - np.maximum(box_min[:, axis, None], cell_start),
                0,
                None,
            )
        )
    is_cell_covered = np.einsum("ra,rb->ab", cell_inside[0], cell_inside[1]) > 0
    covered_area = np.einsum(
        "na,ab,nb->n", cell_overlaps[0], is_cell_covered.astype(float), cell_overlaps[1]
    )
    return covered_area / (boxes[:, 2] * boxes[:, 3])


def add_ignored_coverage_column(table: pl.DataFrame, ignored: pl.DataFrame) -> pl.DataFrame:
    """Add the fraction of each box covered by the ignored regions of its frame.

    Args:
        table: Table with `split`, `image_id`, and pixel box columns `x`, `y`, `w`, `h`.
        ignored: Ignored-region table with `split`, `image_id`, and pixel box columns.

    Returns:
        The table with an `ignored_fraction` column in [0, 1].
    """
    regions_by_frame = {
        key: frame_regions.select(BOX_COLUMNS).to_numpy().astype(np.float64)
        for key, frame_regions in ignored.group_by("split", "image_id")
    }
    no_regions = np.zeros((0, 4))
    frames = []
    for (split, image_id), frame_boxes in table.group_by("split", "image_id", maintain_order=True):
        boxes = frame_boxes.select(BOX_COLUMNS).to_numpy().astype(np.float64)
        regions = regions_by_frame.get((split, image_id), no_regions)
        frames.append(
            frame_boxes.with_columns(
                pl.Series("ignored_fraction", compute_covered_fraction(boxes, regions))
            )
        )
    return pl.concat(frames)


def compute_paired_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute the IoU of corresponding rows of two box arrays.

    Args:
        boxes_a: Array of shape (n, 4) with `x, y, w, h` in px, top-left corner.
        boxes_b: Array of shape (n, 4) in the same format.

    Returns:
        Array of shape (n,) with IoU values in [0, 1].
    """
    overlap = np.clip(
        np.minimum(boxes_a[:, :2] + boxes_a[:, 2:], boxes_b[:, :2] + boxes_b[:, 2:])
        - np.maximum(boxes_a[:, :2], boxes_b[:, :2]),
        0,
        None,
    )
    intersection = overlap[:, 0] * overlap[:, 1]
    union = boxes_a[:, 2] * boxes_a[:, 3] + boxes_b[:, 2] * boxes_b[:, 3] - intersection
    return intersection / union
