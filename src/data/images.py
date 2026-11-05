from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114])


def read_thumbnail(image_path: Path, stride: int) -> np.ndarray:
    """Read an RGB frame and keep every `stride`-th pixel in both directions.

    Args:
        image_path: Path of a JPEG frame.
        stride: Subsampling step in px.

    Returns:
        Array of shape (height / stride, width / stride, 3) with values in [0, 1].
    """
    image = plt.imread(image_path)[::stride, ::stride, :3]
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / np.iinfo(image.dtype).max
    return image.astype(np.float32)


def compute_color_statistics(image_paths: list[str], stride: int) -> pl.DataFrame:
    """Compute mean color, mean luminance, and luminance contrast of frame thumbnails.

    Args:
        image_paths: Paths of JPEG frames.
        stride: Subsampling step in px passed to `read_thumbnail`.

    Returns:
        One row per frame with `file_path`, `mean_r`, `mean_g`, `mean_b`, `luminance_mean`,
        and `luminance_std`, all in [0, 1].
    """
    rows = []
    for image_path in image_paths:
        thumbnail = read_thumbnail(Path(image_path), stride)
        luminance = thumbnail @ LUMINANCE_WEIGHTS
        rows.append(
            (image_path, *thumbnail.mean(axis=(0, 1)).tolist(), luminance.mean(), luminance.std())
        )
    return pl.DataFrame(
        rows,
        schema=["file_path", "mean_r", "mean_g", "mean_b", "luminance_mean", "luminance_std"],
        orient="row",
    )


def compute_frame_differences(frame_pairs: pl.DataFrame, stride: int) -> pl.DataFrame:
    """Compute the mean absolute luminance difference between paired frame thumbnails.

    Args:
        frame_pairs: Table with `file_path` and `file_path_later` columns.
        stride: Subsampling step in px passed to `read_thumbnail`.

    Returns:
        The pair table with a `mean_abs_difference` column in [0, 1].
    """
    differences = [
        float(
            np.abs(
                (read_thumbnail(Path(path), stride) - read_thumbnail(Path(path_later), stride))
                @ LUMINANCE_WEIGHTS
            ).mean()
        )
        for path, path_later in frame_pairs.select("file_path", "file_path_later").iter_rows()
    ]
    return frame_pairs.with_columns(pl.Series("mean_abs_difference", differences))


def sample_frame_pairs(
    frames: pl.DataFrame, offsets: tuple[int, ...], starts_per_clip: int
) -> pl.DataFrame:
    """Pair evenly spaced start frames of every clip with the frames `offset` later.

    Start frames are spread from the first frame to the last frame that still has a partner
    at the largest offset, so every start gets a pair for every offset.

    Args:
        frames: Frame table with `split`, `clip`, `frame_id`, and `file_path`; frame ids of a
            clip are consecutive.
        offsets: Frame distances k to pair.
        starts_per_clip: Number of start frames per clip; shorter clips use fewer.

    Returns:
        One row per pair with `split`, `clip`, `offset`, `frame_id`, `file_path`, and
        `file_path_later`.
    """
    max_offset = max(offsets)
    clip_ranges = frames.group_by("split", "clip").agg(
        pl.col("frame_id").min().alias("first"), pl.col("frame_id").max().alias("last")
    )
    starts = [
        (split, clip, int(start))
        for split, clip, first, last in clip_ranges.sort("split", "clip").iter_rows()
        if last - max_offset >= first
        for start in np.unique(np.round(np.linspace(first, last - max_offset, starts_per_clip)))
    ]
    start_frames = pl.DataFrame(starts, schema=["split", "clip", "frame_id"], orient="row")
    later = frames.select("split", "clip", "frame_id", pl.col("file_path").alias("file_path_later"))
    return (
        start_frames.join(pl.DataFrame({"offset": list(offsets)}), how="cross")
        .join(
            frames.select("split", "clip", "frame_id", "file_path"),
            on=["split", "clip", "frame_id"],
        )
        .join(
            later,
            left_on=["split", "clip", pl.col("frame_id") + pl.col("offset")],
            right_on=["split", "clip", "frame_id"],
        )
        .select("split", "clip", "offset", "frame_id", "file_path", "file_path_later")
        .sort("split", "clip", "offset", "frame_id")
    )
