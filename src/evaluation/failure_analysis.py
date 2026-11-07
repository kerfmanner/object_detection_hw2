from typing import NamedTuple

import numpy as np
import polars as pl

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.boxes import (
    SIZE_BUCKET_TYPE,
    compute_coverage_matrix,
    compute_iou_matrix,
    explode_size_buckets,
)

# Error types and the 0.5 / 0.1 IoU thresholds follow TIDE: https://arxiv.org/abs/2008.08115
MATCH_IOU = 0.5
POOR_LOCALISATION_MIN_IOU = 0.1
IGNORED_MIN_COVERAGE = 0.5
ERROR_CATEGORIES = ("correct", "duplicate", "wrong_class", "poor_localisation", "background")
IGNORED_CATEGORY = "ignored"
MISS_CATEGORY = "miss"
ALL_SIZES = "all"


class DetectionMatches(NamedTuple):
    """Greedy IoU 0.5 matching of all detections of a split to its labelled objects.

    Attributes:
        detections: The predictions ordered by `image_id` and descending `score`, with
            `category` (one of `ERROR_CATEGORIES` or `ignored`), `best_iou` (largest IoU
            with any labelled object of the frame, 0 without objects), and `best_class_id`
            (0-based class of that object, null when no object overlaps).
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple of arrays of shape (n,) with the category, the best IoU, and the index of the
        object with the best IoU (-1 when no object overlaps) of each detection, and an array
        of shape (m,) with the index of the matching detection or -1.
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
    best_object = np.full(len(detection_boxes), -1)
    if len(object_boxes):
        best_object = np.where(best_iou > 0, iou.argmax(axis=1), -1)
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
    return categories, best_iou, best_object, matching_detection


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
        table.join(keys, on=["split", "image_id"], maintain_order="left")
        for table in (tables.objects, tables.ignored)
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
        object_classes = frame_object_rows["class_id"].to_numpy()
        categories, best_iou, best_object, matching_detection = categorize_frame_detections(
            detections.select(BOX_COLUMNS).to_numpy().astype(np.float64).reshape(-1, 4),
            detections["class_id"].to_numpy(),
            frame_object_rows.select(BOX_COLUMNS).to_numpy().astype(np.float64).reshape(-1, 4),
            object_classes,
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
                pl.Series(
                    "best_class_id",
                    [int(object_classes[j]) if j >= 0 else None for j in best_object],
                    dtype=pl.Int64,
                ),
            )
        )
        object_parts.append(
            frame_object_rows.with_columns(
                pl.Series("matched_score", matched_scores, dtype=pl.Float64)
            )
        )
    detection_schema = sorted_predictions.with_columns(
        category=pl.lit(None, pl.String),
        best_iou=pl.lit(None, pl.Float64),
        best_class_id=pl.lit(None, pl.Int64),
    ).clear()
    object_schema = objects.with_columns(matched_score=pl.lit(None, pl.Float64)).clear()
    return DetectionMatches(
        detections=pl.concat([detection_schema, *detection_parts]),
        objects=pl.concat([object_schema, *object_parts]),
    )


def compute_recall_by_size(matches: DetectionMatches) -> pl.DataFrame:
    """Compute the share of labelled objects matched at IoU 0.5, overall and per size bucket.

    Buckets are those of `explode_size_buckets`, as in the evaluator, so an object of exactly
    32² or 96² px² counts in both adjacent buckets.

    Args:
        matches: Result of `match_detections`, without a score threshold.

    Returns:
        One row for `all` objects and one per size bucket in the order small, medium, large
        with `size_bucket`, `objects`, `matched`, and `recall` (NaN for an empty bucket).
    """
    buckets = pl.DataFrame({"size_bucket": [ALL_SIZES, *SIZE_BUCKET_TYPE.categories]})
    objects = matches.objects
    counts = (
        pl.concat(
            [
                objects.with_columns(pl.lit(ALL_SIZES).alias("size_bucket")),
                explode_size_buckets(objects, pl.col("w") * pl.col("h")).with_columns(
                    pl.col("size_bucket").cast(pl.String)
                ),
            ]
        )
        .group_by("size_bucket")
        .agg(
            pl.len().alias("objects"),
            pl.col("matched_score").is_not_null().sum().alias("matched"),
        )
    )
    return (
        buckets.join(counts, on="size_bucket", how="left", maintain_order="left")
        .with_columns(pl.col("objects", "matched").fill_null(0).cast(pl.Int64))
        .with_columns((pl.col("matched") / pl.col("objects")).alias("recall"))
    )


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


def select_error_examples(
    matches: DetectionMatches,
    images: pl.DataFrame,
    threshold: float,
    per_category: int,
    seed: int,
) -> pl.DataFrame:
    """Select example detections of every error category and example missed objects.

    For each category of `ERROR_CATEGORIES`, the highest-scoring detections at or above the
    threshold are taken, at most one per clip, so that consecutive frames of one object do
    not fill a category. Missed objects (unmatched at the threshold) are drawn at random with
    the seed, also at most one per clip.

    Args:
        matches: Result of `match_detections`.
        images: Image table with `image_id` and `clip` of the matched frames.
        threshold: Lowest kept score, e.g. from `find_best_f1_threshold`.
        per_category: Examples per category, fewer when a category has fewer frames.
        seed: Seed of the draw of missed objects.

    Returns:
        Rows in the order of `ERROR_CATEGORIES` and then `miss` with `category`, `image_id`,
        0-based `class_id`, `score` (null for misses), and pixel box columns.
    """
    columns = ["category", "image_id", "class_id", "score", *BOX_COLUMNS]
    clips = images.select("image_id", "clip")
    detections = (
        matches.detections.filter(
            pl.col("score") >= threshold, pl.col("category").is_in(ERROR_CATEGORIES)
        )
        .join(clips, on="image_id", maintain_order="left")
        .sort("score", descending=True, maintain_order=True)
        .unique(subset=["category", "clip"], keep="first", maintain_order=True)
        .group_by("category", maintain_order=True)
        .head(per_category)
        .sort(pl.col("category").cast(pl.Enum(ERROR_CATEGORIES)), maintain_order=True)
        .select(columns)
    )
    missed = (
        matches.objects.filter(
            pl.col("matched_score").is_null() | (pl.col("matched_score") < threshold)
        )
        .join(clips, on="image_id", maintain_order="left")
        .sort("image_id", "class_id", *BOX_COLUMNS)
        .sample(fraction=1.0, shuffle=True, seed=seed)
    )
    misses = (
        missed.unique(subset="clip", keep="first", maintain_order=True)
        .head(per_category)
        .select(
            pl.lit(MISS_CATEGORY).alias("category"),
            "image_id",
            "class_id",
            pl.lit(None, pl.Float64).alias("score"),
            *BOX_COLUMNS,
        )
    )
    return pl.concat([detections, misses.cast(detections.schema)])


def compute_proposal_recall(
    objects: pl.DataFrame,
    proposals: pl.DataFrame,
    top_ns: list[int],
    iou_thresholds: list[float],
) -> pl.DataFrame:
    """Compute the share of labelled objects covered by the top proposals of their frame.

    An object is covered by the top n proposals if one of them has an IoU of at least the
    threshold with it, regardless of class, the proposal recall of the Faster R-CNN paper
    (https://arxiv.org/abs/1506.01497, section 4.1).

    Args:
        objects: Labelled objects of the analysed frames with `image_id` and pixel box
            columns.
        proposals: Proposals with `image_id`, `rank` (0 for the highest objectness), and
            pixel box columns; frames without rows have no proposals.
        top_ns: Numbers of top proposals per frame.
        iou_thresholds: IoU thresholds.

    Returns:
        One row per IoU threshold, n, and size bucket (`all`, then the COCO buckets as in
        `explode_size_buckets`) with `iou_threshold`, `top_n`, `size_bucket`, `objects`,
        and `recall`.
    """
    frame_proposals = proposals.sort("rank").partition_by("image_id", as_dict=True)
    parts = []
    for (image_id,), frame_objects in objects.partition_by("image_id", as_dict=True).items():
        boxes = frame_proposals.get((image_id,), proposals.clear()).select(BOX_COLUMNS)
        iou = compute_iou_matrix(
            frame_objects.select(BOX_COLUMNS).to_numpy().astype(np.float64),
            boxes.to_numpy().astype(np.float64).reshape(-1, 4),
        )
        for iou_threshold in iou_thresholds:
            is_covering = iou >= iou_threshold
            first_rank = np.full(frame_objects.height, np.inf)
            if boxes.height:
                first_rank = np.where(is_covering.any(axis=1), is_covering.argmax(axis=1), np.inf)
            parts.append(
                frame_objects.select(BOX_COLUMNS).with_columns(
                    pl.lit(iou_threshold).alias("iou_threshold"),
                    pl.Series("first_rank", first_rank, dtype=pl.Float64),
                )
            )
    ranked = pl.concat(parts)
    bucketed = pl.concat(
        [
            ranked.with_columns(pl.lit(ALL_SIZES).alias("size_bucket")),
            explode_size_buckets(ranked, pl.col("w") * pl.col("h")).with_columns(
                pl.col("size_bucket").cast(pl.String)
            ),
        ]
    )
    order = pl.Enum([ALL_SIZES, *SIZE_BUCKET_TYPE.categories])
    return pl.concat(
        bucketed.group_by("iou_threshold", "size_bucket").agg(
            pl.lit(top_n).alias("top_n"),
            pl.len().alias("objects"),
            (pl.col("first_rank") < top_n).mean().alias("recall"),
        )
        for top_n in top_ns
    ).select(
        pl.col("iou_threshold", "top_n", "size_bucket", "objects", "recall").sort_by(
            "iou_threshold", "top_n", pl.col("size_bucket").cast(order)
        )
    )
