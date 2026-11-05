from pathlib import Path

import polars as pl

from src.evaluation.coco_metrics import MAX_DETECTIONS_PER_IMAGE, find_invalid_prediction_rows

SUBMISSION_COLUMNS = ("image_id", "PredictionString")
COORDINATE_DECIMALS = 2
SCORE_DECIMALS = 5


def format_submission(predictions: pl.DataFrame, test_images: pl.DataFrame) -> pl.DataFrame:
    """Convert per-frame predictions into Kaggle submission rows.

    Boxes are clipped to their frame by rounding and clipping the corners, so `x + w` and
    `y + h` never exceed the frame size; boxes that become empty are dropped. Each frame then
    keeps its 500 highest-scoring boxes in descending score order.

    Args:
        predictions: Table with `image_id` from `test_images`, 0-based `class_id`, `score`
            in [0, 1], and pixel box `x`, `y`, `w`, `h` (top-left corner, `w`, `h` > 0).
        test_images: Table with `image_id`, `width`, and `height` of every test frame, e.g.
            from `load_test_images`.

    Returns:
        One row per test frame in the order of `test_images` with `image_id` and
        `PredictionString` (`class score x y w h` per box, separated by spaces; empty
        string for frames without boxes).
    """
    invalid = find_invalid_prediction_rows(predictions, test_images)
    if invalid:
        raise ValueError(f"Invalid predictions (rows per problem): {invalid}")
    x1 = pl.col("x").clip(0, pl.col("width")).round(COORDINATE_DECIMALS)
    y1 = pl.col("y").clip(0, pl.col("height")).round(COORDINATE_DECIMALS)
    x2 = (pl.col("x") + pl.col("w")).clip(0, pl.col("width")).round(COORDINATE_DECIMALS)
    y2 = (pl.col("y") + pl.col("h")).clip(0, pl.col("height")).round(COORDINATE_DECIMALS)
    boxes = (
        predictions.join(test_images.select("image_id", "width", "height"), on="image_id")
        .select(
            "image_id",
            "class_id",
            pl.col("score").round(SCORE_DECIMALS),
            x1.alias("x"),
            y1.alias("y"),
            (x2 - x1).round(COORDINATE_DECIMALS).alias("w"),
            (y2 - y1).round(COORDINATE_DECIMALS).alias("h"),
        )
        .filter(pl.col("w") > 0, pl.col("h") > 0)
        .sort("image_id", "score", descending=[False, True], maintain_order=True)
        .group_by("image_id", maintain_order=True)
        .head(MAX_DETECTIONS_PER_IMAGE)
        .group_by("image_id", maintain_order=True)
        .agg(
            pl.concat_str(
                pl.col("class_id", "score", "x", "y", "w", "h").cast(pl.String), separator=" "
            )
            .str.join(" ")
            .alias("PredictionString")
        )
    )
    return (
        test_images.select("image_id")
        .join(boxes, on="image_id", how="left", maintain_order="left")
        .with_columns(pl.col("PredictionString").fill_null(""))
    )


def write_submission(
    predictions: pl.DataFrame, test_images: pl.DataFrame, csv_path: Path
) -> pl.DataFrame:
    """Write predictions as a Kaggle submission CSV.

    Args:
        predictions: Table accepted by `format_submission`.
        test_images: Table with `image_id`, `width`, and `height` of every test frame.
        csv_path: Output path of the CSV file.

    Returns:
        The written submission rows.
    """
    submission = format_submission(predictions, test_images)
    submission.write_csv(csv_path)
    return submission
