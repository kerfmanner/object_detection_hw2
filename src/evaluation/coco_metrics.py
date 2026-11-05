import logging

import numpy as np
import polars as pl
from faster_coco_eval import COCO, COCOeval_faster

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.classes import CLASS_NAMES

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = ("image_id", "class_id", "score", *BOX_COLUMNS)
MAX_DETECTIONS_PER_IMAGE = 500
MAX_TEST_WIDTH = 1920
CATEGORIES = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, start=1)]


def validate_predictions(predictions: pl.DataFrame, images: pl.DataFrame) -> None:
    """Check that predictions refer to evaluated frames and contain valid boxes.

    Args:
        predictions: Table with `image_id`, 0-based `class_id`, `score`, and pixel box
            `x`, `y`, `w`, `h` (top-left corner).
        images: Image table of the evaluated frames with `image_id`.
    """
    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Predictions lack columns {missing_columns}")
    problems = {
        "unknown image_id": ~pl.col("image_id").is_in(images["image_id"].implode()),
        "class_id outside 0-7": ~pl.col("class_id").is_between(0, len(CLASS_NAMES) - 1),
        "score outside [0, 1]": ~pl.col("score").is_between(0, 1),
        "non-positive w or h": (pl.col("w") <= 0) | (pl.col("h") <= 0),
        "null value": pl.any_horizontal(pl.col(PREDICTION_COLUMNS).is_null()),
        "non-finite value": pl.any_horizontal(~pl.col("score", *BOX_COLUMNS).is_finite()),
    }
    counts = predictions.select(
        expression.sum().alias(problem) for problem, expression in problems.items()
    ).row(0, named=True)
    crowded_frames = (
        predictions.group_by("image_id").len().filter(pl.col("len") > MAX_DETECTIONS_PER_IMAGE)
    )
    found = {problem: count for problem, count in counts.items() if count}
    if crowded_frames.height:
        found[f"frames with more than {MAX_DETECTIONS_PER_IMAGE} boxes"] = crowded_frames.height
    if found:
        raise ValueError(f"Invalid predictions (rows or frames per problem): {found}")


def build_coco_annotations(boxes: pl.DataFrame) -> list[dict]:
    """Convert a box table into COCO annotations, mapping 0-based `class_id` to `category_id`.

    This is the only place where class ids become 1-based COCO category ids.

    Args:
        boxes: Table with `image_id`, 0-based `class_id`, `iscrowd`, pixel box `x`, `y`, `w`,
            `h`, and optionally `score`.

    Returns:
        COCO annotation dictionaries with ids from 1 and `area` = w * h.
    """
    score = ["score"] if "score" in boxes.columns else []
    return (
        boxes.with_row_index("id", offset=1)
        .select(
            "id",
            "image_id",
            (pl.col("class_id") + 1).alias("category_id"),
            pl.concat_list(pl.col(BOX_COLUMNS).cast(pl.Float64)).alias("bbox"),
            (pl.col("w") * pl.col("h")).cast(pl.Float64).alias("area"),
            "iscrowd",
            *score,
        )
        .to_dicts()
    )


def build_coco_ground_truth(tables: AnnotationTables) -> dict:
    """Build a COCO ground truth dictionary from the tables of one split.

    Ignored regions become `iscrowd=1` annotations repeated once per class, so COCO ignores
    detections of any class that lie inside them.

    Args:
        tables: Annotation tables restricted to the evaluated frames of one split.

    Returns:
        COCO dictionary with `images`, `annotations`, and `categories` keys.
    """
    class_ids = pl.DataFrame({"class_id": range(len(CLASS_NAMES))})
    columns = ["image_id", "class_id", *BOX_COLUMNS, "iscrowd"]
    objects = tables.objects.with_columns(iscrowd=pl.lit(0)).select(columns)
    crowd = tables.ignored.join(class_ids, how="cross").with_columns(iscrowd=pl.lit(1))
    boxes = pl.concat([objects, crowd.select(columns)])
    return {
        "images": tables.images.select(
            pl.col("image_id").alias("id"), "width", "height"
        ).to_dicts(),
        "annotations": build_coco_annotations(boxes),
        "categories": CATEGORIES,
    }


def compute_average_precision(
    evaluator: COCOeval_faster,
    iou_threshold: float | None,
    area: str,
    class_id: int | None,
) -> float | None:
    """Average the accumulated precision like `COCOeval.summarize` at the largest maxDets.

    Follows `_summarize` of pycocotools:
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        evaluator: Evaluator after `evaluate` and `accumulate`.
        iou_threshold: Single IoU threshold, or None to average over 0.50:0.95.
        area: Area range label, `all`, `small`, `medium`, or `large`.
        class_id: 0-based class to report, or None to average over classes.

    Returns:
        Average precision in [0, 1], or None when no ground truth falls in the selection.
    """
    params = evaluator.params
    precision = evaluator.eval["precision"][..., params.areaRngLbl.index(area), -1]
    if iou_threshold is not None:
        precision = precision[np.isclose(params.iouThrs, iou_threshold)]
    if class_id is not None:
        precision = precision[:, :, class_id]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else None


def run_coco_evaluation(
    ground_truth: COCO, detections: COCO, image_ids: list[int]
) -> COCOeval_faster:
    """Run bbox evaluation with up to 500 detections per frame.

    Size buckets are the evaluator's standard COCO ranges (`src.data.boxes.SIZE_RANGES`)
    with inclusive ends, so a box of exactly 32² or 96² px² counts in both adjacent buckets.

    Args:
        ground_truth: COCO ground truth with `iscrowd=1` ignored regions.
        detections: COCO detections on the same images and categories.
        image_ids: Frames to evaluate.

    Returns:
        Evaluator after `evaluate` and `accumulate`.
    """
    evaluator = COCOeval_faster(ground_truth, detections, "bbox", print_function=logger.debug)
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, MAX_DETECTIONS_PER_IMAGE]
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def evaluate_detections(tables: AnnotationTables, split: str, predictions: pl.DataFrame) -> dict:
    """Compute COCO metrics of predictions on all frames of a split present in `tables`.

    Args:
        tables: Annotation tables; every frame of `split` in `tables.images` is evaluated.
        split: Split name, e.g. `val`.
        predictions: Table with `image_id`, 0-based `class_id`, `score`, and pixel box
            `x`, `y`, `w`, `h` (top-left corner); frames without rows have no detections.

    Returns:
        Dictionary with `frames`, `objects`, `map`, `map50`, `map75`, `ap_small`,
        `ap_medium`, `ap_large`, `per_class_ap` (class name to AP), and `frames`, `map`, and
        `ap_small` on frames at most 1920 px wide with the suffix `_max_width_1920`.
        AP values are None where no ground truth exists.
    """
    images = tables.images.filter(pl.col("split") == split)
    validate_predictions(predictions, images)
    keys = images.select("split", "image_id")
    split_tables = AnnotationTables(
        images=images,
        objects=tables.objects.join(keys, on=["split", "image_id"]),
        ignored=tables.ignored.join(keys, on=["split", "image_id"]),
    )
    ground_truth = COCO(build_coco_ground_truth(split_tables), print_function=logger.debug)
    detection_dataset = {
        "images": [dict(image) for image in ground_truth.dataset["images"]],
        "annotations": build_coco_annotations(predictions.with_columns(iscrowd=pl.lit(0))),
        "categories": CATEGORIES,
    }
    detections = COCO(detection_dataset, print_function=logger.debug)
    evaluator = run_coco_evaluation(ground_truth, detections, images["image_id"].to_list())
    narrow_ids = images.filter(pl.col("width") <= MAX_TEST_WIDTH)["image_id"].to_list()
    narrow = run_coco_evaluation(ground_truth, detections, narrow_ids) if narrow_ids else None
    return {
        "frames": images.height,
        "objects": split_tables.objects.height,
        "map": compute_average_precision(evaluator, None, "all", None),
        "map50": compute_average_precision(evaluator, 0.5, "all", None),
        "map75": compute_average_precision(evaluator, 0.75, "all", None),
        "ap_small": compute_average_precision(evaluator, None, "small", None),
        "ap_medium": compute_average_precision(evaluator, None, "medium", None),
        "ap_large": compute_average_precision(evaluator, None, "large", None),
        "per_class_ap": {
            name: compute_average_precision(evaluator, None, "all", class_id)
            for class_id, name in enumerate(CLASS_NAMES)
        },
        f"frames_max_width_{MAX_TEST_WIDTH}": len(narrow_ids),
        f"map_max_width_{MAX_TEST_WIDTH}": (
            compute_average_precision(narrow, None, "all", None) if narrow else None
        ),
        f"ap_small_max_width_{MAX_TEST_WIDTH}": (
            compute_average_precision(narrow, None, "small", None) if narrow else None
        ),
    }
