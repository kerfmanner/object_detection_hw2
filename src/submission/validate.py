import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from decouple import config

from src.data.annotations import load_test_images
from src.data.classes import CLASS_NAMES
from src.evaluation.coco_metrics import MAX_DETECTIONS_PER_IMAGE
from src.submission.writer import SUBMISSION_COLUMNS

logger = logging.getLogger(__name__)

BOX_FIELDS = ("class_id", "score", "x", "y", "w", "h")
FRAME_TOLERANCE = 1e-6


def parse_submission(csv_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read a submission CSV and split its prediction strings into boxes.

    Args:
        csv_path: Path of a CSV with the header `image_id,PredictionString`.

    Returns:
        The rows with `image_id` and `PredictionString` (empty string for empty frames), and
        one row per box with `image_id`, `box`, `token_count` (tokens of its frame), and
        string fields `class_id`, `score`, `x`, `y`, `w`, `h` (null if the frame's tokens
        do not fill the box).
    """
    with csv_path.open() as csv_file:
        header = tuple(csv_file.readline().strip().split(","))
    if header != SUBMISSION_COLUMNS:
        raise ValueError(f"{csv_path} has header {header}, expected {SUBMISSION_COLUMNS}")
    rows = pl.read_csv(
        csv_path, schema={column: pl.String for column in SUBMISSION_COLUMNS}
    ).with_columns(pl.col("PredictionString").fill_null(""))
    tokens = (
        rows.select("image_id", pl.col("PredictionString").str.split(" ").alias("token"))
        .explode("token")
        .filter(pl.col("token") != "")
        .with_columns(pl.int_range(pl.len()).over("image_id").alias("position"))
        .with_columns(
            (pl.col("position") // len(BOX_FIELDS)).alias("box"),
            pl.len().over("image_id").alias("token_count"),
        )
    )
    boxes = (
        tokens.group_by("image_id", "box", "token_count", maintain_order=True)
        .agg(pl.col("token"))
        .select(
            "image_id",
            "box",
            "token_count",
            *(
                pl.col("token").list.get(i, null_on_oob=True).alias(field)
                for i, field in enumerate(BOX_FIELDS)
            ),
        )
    )
    return rows, boxes


def validate_submission(csv_path: Path, test_images: pl.DataFrame) -> int:
    """Check a submission CSV against the competition format and the test frames.

    Args:
        csv_path: Path of the submission CSV.
        test_images: Table with string `image_id`, `width`, and `height` of every test frame.

    Returns:
        Number of boxes in the submission.
    """
    rows, boxes = parse_submission(csv_path)
    found = {
        "duplicate image_id rows": rows.height - rows["image_id"].n_unique(),
        "missing test image_id": test_images.join(rows, on="image_id", how="anti").height,
        "unknown image_id": rows.join(test_images, on="image_id", how="anti").height,
        "frames whose token count is not a multiple of 6": boxes.filter(
            pl.col("token_count") % len(BOX_FIELDS) != 0
        )["image_id"].n_unique(),
        f"frames with more than {MAX_DETECTIONS_PER_IMAGE} boxes": boxes.group_by("image_id")
        .len()
        .filter(pl.col("len") > MAX_DETECTIONS_PER_IMAGE)
        .height,
    }
    values = (
        boxes.filter(pl.col("token_count") % len(BOX_FIELDS) == 0)
        .join(test_images.select("image_id", "width", "height"), on="image_id")
        .select(
            pl.col("class_id").cast(pl.Int64, strict=False),
            *(pl.col(field).cast(pl.Float64, strict=False) for field in BOX_FIELDS[1:]),
            "width",
            "height",
        )
    )
    problems = {
        "non-integer class": pl.col("class_id").is_null(),
        "unparsable number": pl.any_horizontal(pl.col(BOX_FIELDS[1:]).is_null()),
        "class outside 0-7": ~pl.col("class_id").is_between(0, len(CLASS_NAMES) - 1),
        "score outside [0, 1]": ~pl.col("score").is_between(0, 1),
        "non-positive w or h": (pl.col("w") <= 0) | (pl.col("h") <= 0),
        "box outside its frame": (pl.col("x") < 0)
        | (pl.col("y") < 0)
        | (pl.col("x") + pl.col("w") > pl.col("width") + FRAME_TOLERANCE)
        | (pl.col("y") + pl.col("h") > pl.col("height") + FRAME_TOLERANCE),
        "non-finite value": pl.any_horizontal(~pl.col(BOX_FIELDS[1:]).is_finite()),
    }
    found |= values.select(
        expression.fill_null(False).sum().alias(problem) for problem, expression in problems.items()
    ).row(0, named=True)
    found = {problem: count for problem, count in found.items() if count}
    if found:
        raise ValueError(f"Invalid submission {csv_path} (rows, frames, or boxes): {found}")
    return boxes.height


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission CSV.")
    parser.add_argument("csv", type=Path, help="Submission CSV to validate.")
    args = parser.parse_args()
    test_images = load_test_images(Path(config("DATA_ROOT")))
    box_count = validate_submission(args.csv, test_images)
    logger.info("%s is valid: %d frames, %d boxes", args.csv, test_images.height, box_count)


if __name__ == "__main__":
    main()
