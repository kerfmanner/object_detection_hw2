from typing import NamedTuple

import numpy as np
import polars as pl

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.boxes import compute_coverage_matrix, compute_iou_matrix
from src.evaluation.coco_metrics import SIZE_RANGES

# Error types and the 0.5 / 0.1 IoU thresholds follow TIDE: https://arxiv.org/abs/2008.08115
MATCH_IOU = 0.5
POOR_LOCALISATION_MIN_IOU = 0.1
IGNORED_MIN_COVERAGE = 0.5
ERROR_CATEGORIES = ("correct", "duplicate", "wrong_class", "poor_localisation", "background")
IGNORED_CATEGORY = "ignored"
MISS_CATEGORY = "miss"


class DetectionMatches(NamedTuple):
    """Greedy IoU 0.5 matching of all detections of a split to its labelled objects.

    Attributes:
        detections: The predictions ordered by `image_id` and descending `score`, with
            `category` (one of `ERROR_CATEGORIES` or `ignored`) and `best_iou` (largest IoU
            with any labelled object of the frame, 0 without objects).
        objects: The labelled objects of the evaluated frames with `matched_score` (score of
            the matching detection, null when no detection matches).
    """

    detections: pl.DataFrame
    objects: pl.DataFrame


def categorize_frame_detections(
    detection_boxes: np.ndarray,
    detection_classes: np.ndarray,
    object_boxes: np.ndarray,
    object_classes: np.ndarray,
    regions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the detections of one frame greedily and assign each an error category.

    Detections are visited in the given order (descending score). A detection is `correct`
    if it has IoU ≥ 0.5 with an unmatched object of its class and then matches the one with
    the largest IoU. Otherwise it is `ignored` if at least half of its area lies inside one
    ignored region, `duplicate` if it has IoU ≥ 0.5 with an already matched object of its
    class, `wrong_class` if it has IoU ≥ 0.5 only with objects of other classes,
    `poor_localisation` if its best IoU with any object is in [0.1, 0.5), and `background`
    below 0.1.

    Args:
        detection_boxes: Array of shape (n, 4) with `x, y, w, h` in px, sorted by score.
        detection_classes: Array of shape (n,) with 0-based class ids.
        object_boxes: Array of shape (m, 4) with `x, y, w, h` in px.
        object_classes: Array of shape (m,) with 0-based class ids.
        regions: Array of shape (k, 4) with the ignored regions in px.

    Returns:
        Tuple of arrays of shape (n,) with the category and the best IoU of each detection,
        and an array of shape (m,) with the index of the matching detection or -1.
    """
    iou = compute_iou_matrix(detection_boxes, object_boxes)
    is_same_class = detection_classes[:, None] == object_classes[None, :]
    matching_detection = np.full(len(object_boxes), -1)
    is_correct = np.zeros(len(detection_boxes), dtype=bool)
    for i in range(len(detection_boxes)):
        candidates = np.where(is_same_class[i] & (matching_detection < 0), iou[i], -1.0)
        if len(candidates) and candidates.max() >= MATCH_IOU:
            matching_detection[candidates.argmax()] = i
            is_correct[i] = True
    coverage = compute_coverage_matrix(detection_boxes, regions)
    is_ignored = coverage.max(axis=1, initial=0.0) >= IGNORED_MIN_COVERAGE
    best_iou = iou.max(axis=1, initial=0.0)
    categories = np.select(
        [
            is_correct,
            is_ignored,
            np.where(is_same_class, iou, 0.0).max(axis=1, initial=0.0) >= MATCH_IOU,
            np.where(is_same_class, 0.0, iou).max(axis=1, initial=0.0) >= MATCH_IOU,
            best_iou >= POOR_LOCALISATION_MIN_IOU,
        ],
        ["correct", IGNORED_CATEGORY, "duplicate", "wrong_class", "poor_localisation"],
        default="background",
    )
    return categories, best_iou, matching_detection


def match_detections(
    tables: AnnotationTables, split: str, predictions: pl.DataFrame
) -> DetectionMatches:
    """Match the predictions of all frames of a split present in `tables` to their objects.

    Args:
        tables: Annotation tables; every frame of `split` in `tables.images` is evaluated.
        split: Split name, e.g. `val`.
        predictions: Table with `image_id`, 0-based `class_id`, `score`, and pixel box
            `x`, `y`, `w`, `h`, as accepted by `evaluate_detections`.

    Returns:
        Categorized detections and objects with their matching scores.
    """
    keys = tables.images.filter(pl.col("split") == split).select("split", "image_id")
    objects, ignored = (
        table.join(keys, on=["split", "image_id"]) for table in (tables.objects, tables.ignored)
    )
    sorted_predictions = predictions.sort(
        "image_id", pl.col("score"), descending=[False, True], maintain_order=True
    )
    frame_objects = objects.partition_by("image_id", as_dict=True)
    frame_regions = ignored.partition_by("image_id", as_dict=True)
    frame_predictions = sorted_predictions.partition_by("image_id", as_dict=True)
    empty_objects = objects.clear()
    empty_regions = ignored.clear()
    detection_parts = []
    object_parts = []
    for (image_id,) in keys.select("image_id").iter_rows():
        detections = frame_predictions.get((image_id,), sorted_predictions.clear())
        frame_object_rows = frame_objects.get((image_id,), empty_objects)
        categories, best_iou, matching_detection = categorize_frame_detections(
            detections.select(BOX_COLUMNS).to_numpy().astype(np.float64).reshape(-1, 4),
            detections["class_id"].to_numpy(),
            frame_object_rows.select(BOX_COLUMNS).to_numpy().astype(np.float64).reshape(-1, 4),
            frame_object_rows["class_id"].to_numpy(),
            frame_regions.get((image_id,), empty_regions)
            .select(BOX_COLUMNS)
            .to_numpy()
            .astype(np.float64)
            .reshape(-1, 4),
        )
        scores = detections["score"].to_numpy()
        matched_scores = [scores[i] if i >= 0 else None for i in matching_detection]
        detection_parts.append(
            detections.with_columns(
                pl.Series("category", categories, dtype=pl.String),
                pl.Series("best_iou", best_iou, dtype=pl.Float64),
            )
        )
        object_parts.append(
            frame_object_rows.with_columns(
                pl.Series("matched_score", matched_scores, dtype=pl.Float64)
            )
        )
    detection_schema = sorted_predictions.with_columns(
        category=pl.lit(None, pl.String), best_iou=pl.lit(None, pl.Float64)
    ).clear()
    object_schema = objects.with_columns(matched_score=pl.lit(None, pl.Float64)).clear()
    return DetectionMatches(
        detections=pl.concat([detection_schema, *detection_parts]),
        objects=pl.concat([object_schema, *object_parts]),
    )


def compute_recall_by_size(matches: DetectionMatches) -> pl.DataFrame:
    """Compute the share of labelled objects matched at IoU 0.5 per COCO size bucket.

    Buckets are the inclusive ranges of `SIZE_RANGES` used by the evaluator, so an object of
    exactly 32² or 96² px² counts in both adjacent buckets.

    Args:
        matches: Result of `match_detections`, without a score threshold.

    Returns:
        One row per size bucket with `size`, `objects`, `matched`, and `recall` (NaN for an
        empty bucket).
    """
    area = pl.col("w") * pl.col("h")
    return pl.concat(
        matches.objects.filter(area.is_between(low, high)).select(
            pl.lit(size).alias("size"),
            pl.len().cast(pl.Int64).alias("objects"),
            pl.col("matched_score").is_not_null().sum().cast(pl.Int64).alias("matched"),
        )
        for size, (low, high) in SIZE_RANGES.items()
    ).with_columns((pl.col("matched") / pl.col("objects")).alias("recall"))


def find_best_f1_threshold(matches: DetectionMatches) -> dict[str, float | None]:
    """Find the score threshold that maximizes F1 at IoU 0.5 over all classes.

    Detections inside ignored regions are neither true nor false positives. Greedy matching
    in descending score order makes the categories of the detections above any threshold
    equal to those of matching only them. Ties in F1 go to the highest threshold.

    Args:
        matches: Result of `match_detections`.

    Returns:
        Dictionary with `threshold` (lowest kept score, None without detections),
        `precision`, `recall`, and `f1`.
    """
    object_count = matches.objects.height
    curve = (
        matches.detections.filter(pl.col("category") != IGNORED_CATEGORY)
        .sort("score", descending=True)
        .select(
            "score",
            (pl.col("category") == "correct").cum_sum().alias("true_positives"),
            pl.int_range(1, pl.len() + 1).alias("detections"),
        )
        .filter(pl.col("score") != pl.col("score").shift(-1).fill_null(-1.0))
        .with_columns(
            (pl.col("true_positives") / pl.col("detections")).alias("precision"),
            (pl.col("true_positives") / max(object_count, 1)).alias("recall"),
        )
        .with_columns(
            (2 * pl.col("true_positives") / (pl.col("detections") + object_count)).alias("f1")
        )
    )
    if curve.is_empty():
        return {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best = curve.row(int(curve["f1"].arg_max() or 0), named=True)
    return {
        "threshold": best["score"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
    }


def count_error_categories(matches: DetectionMatches, threshold: float) -> pl.DataFrame:
    """Count detections per error category and missed objects at a score threshold.

    Args:
        matches: Result of `match_detections`.
        threshold: Lowest kept score, e.g. from `find_best_f1_threshold`.

    Returns:
        One row per category of `ERROR_CATEGORIES` plus `miss` with `category` and `count`;
        detections inside ignored regions are excluded.
    """
    kept = matches.detections.filter(pl.col("score") >= threshold)
    counts = dict(kept.group_by("category").len().iter_rows())
    misses = matches.objects.filter(
        pl.col("matched_score").is_null() | (pl.col("matched_score") < threshold)
    ).height
    return pl.DataFrame(
        {
            "category": [*ERROR_CATEGORIES, MISS_CATEGORY],
            "count": [*(counts.get(category, 0) for category in ERROR_CATEGORIES), misses],
        }
    )
