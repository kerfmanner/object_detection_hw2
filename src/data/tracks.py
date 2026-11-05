import numpy as np
import polars as pl

from src.data.annotations import BOX_COLUMNS
from src.data.boxes import compute_paired_iou

TRACK_KEY = ["split", "clip", "track_id"]


def pair_track_boxes(tracked_objects: pl.DataFrame, offset: int) -> pl.DataFrame:
    """Pair every object with the same track `offset` frames later and measure the change.

    Args:
        tracked_objects: Object table with `split`, `clip`, `track_id`, `frame_id`, `class_id`,
            and pixel box columns.
        offset: Frame distance k between the two boxes of a pair.

    Returns:
        One row per pair present in both frames with `split`, `clip`, `track_id`, `class_id`
        at frame t, `offset`, center `displacement` in px, `relative_displacement` (divided by
        the square root of the box area at t), and `iou` between the two boxes.
    """
    current = tracked_objects.select(*TRACK_KEY, "frame_id", "class_id", *BOX_COLUMNS)
    later = tracked_objects.select(
        *TRACK_KEY,
        (pl.col("frame_id") - offset).alias("frame_id"),
        *(pl.col(column).alias(f"{column}_later") for column in BOX_COLUMNS),
    )
    pairs = current.join(later, on=[*TRACK_KEY, "frame_id"])
    boxes = pairs.select(BOX_COLUMNS).to_numpy().astype(np.float64)
    boxes_later = pairs.select(f"{c}_later" for c in BOX_COLUMNS).to_numpy().astype(np.float64)
    center_shift = (boxes_later[:, :2] + boxes_later[:, 2:] / 2) - (boxes[:, :2] + boxes[:, 2:] / 2)
    displacement = np.hypot(center_shift[:, 0], center_shift[:, 1])
    return pairs.select(
        *TRACK_KEY,
        "class_id",
        pl.lit(offset).alias("offset"),
        pl.Series("displacement", displacement),
        pl.Series("relative_displacement", displacement / np.sqrt(boxes[:, 2] * boxes[:, 3])),
        pl.Series("iou", compute_paired_iou(boxes, boxes_later)),
    )


def summarize_tracks(tracked_objects: pl.DataFrame) -> pl.DataFrame:
    """Describe every track by its class, length, and frame span.

    Args:
        tracked_objects: Object table with `split`, `clip`, `track_id`, `frame_id`, and
            `class_id`.

    Returns:
        One row per track with the most frequent `class_id`, `class_count` (distinct classes),
        `frames` (annotated frames), `span` (last minus first frame plus one), and `has_gap`.
    """
    return (
        tracked_objects.group_by(TRACK_KEY)
        .agg(
            pl.col("class_id").mode().min().alias("class_id"),
            pl.col("class_id").n_unique().alias("class_count"),
            pl.len().alias("frames"),
            (pl.col("frame_id").max() - pl.col("frame_id").min() + 1).alias("span"),
        )
        .with_columns((pl.col("span") > pl.col("frames")).alias("has_gap"))
        .sort(TRACK_KEY)
    )
