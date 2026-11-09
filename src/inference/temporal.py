import numpy as np
import polars as pl

from src.data.annotations import BOX_COLUMNS
from src.data.boxes import compute_iou_matrix
from src.evaluation.coco_metrics import PREDICTION_COLUMNS

TRACK_STATISTICS = ("mean", "max")


def match_greedily(ious: np.ndarray, link_iou: float) -> tuple[np.ndarray, np.ndarray]:
    """Match the boxes of two frames one to one, highest IoU first.

    Args:
        ious: Array (n, m) of IoU between the boxes of a frame and of the next frame.
        link_iou: Pairs need at least this IoU.

    Returns:
        Row indices into the first frame and column indices into the second, of equal length.
    """
    rows, columns = np.nonzero(ious >= link_iou)
    order = np.argsort(-ious[rows, columns], kind="stable")
    is_row_free = np.ones(ious.shape[0], dtype=bool)
    is_column_free = np.ones(ious.shape[1], dtype=bool)
    matched_rows, matched_columns = [], []
    for row, column in zip(rows[order], columns[order], strict=True):
        if is_row_free[row] and is_column_free[column]:
            is_row_free[row] = is_column_free[column] = False
            matched_rows.append(row)
            matched_columns.append(column)
    return np.array(matched_rows, dtype=np.int64), np.array(matched_columns, dtype=np.int64)


def link_track_ids(frame_ids: np.ndarray, boxes: np.ndarray, link_iou: float) -> np.ndarray:
    """Chain the detections of one class in one clip into tracks across consecutive frames.

    The detections of frame t are matched to those of frame t + 1 with `match_greedily`; a
    matched detection continues the track of its predecessor, every other detection starts a
    new track. A frame without detections ends all tracks.

    Args:
        frame_ids: Array (n,) of frame numbers, sorted ascending.
        boxes: Array (n, 4) with `x, y, w, h` in px, in the order of `frame_ids`.
        link_iou: Minimum IoU of a link between consecutive frames.

    Returns:
        Array (n,) of track ids from 0, local to this clip and class.
    """
    track_ids = np.arange(len(frame_ids))
    starts = np.flatnonzero(np.diff(frame_ids, prepend=frame_ids[:1] - 1))
    bounds = list(zip(starts, [*starts[1:], len(frame_ids)], strict=True))
    for (start, middle), (_, end) in zip(bounds, bounds[1:], strict=False):
        if frame_ids[middle] != frame_ids[start] + 1:
            continue
        rows, columns = match_greedily(
            compute_iou_matrix(boxes[start:middle], boxes[middle:end]), link_iou
        )
        track_ids[middle + columns] = track_ids[start + rows]
    return np.unique(track_ids, return_inverse=True)[1]


def assign_tracks(
    predictions: pl.DataFrame, images: pl.DataFrame, link_iou: float, min_score: float
) -> pl.DataFrame:
    """Link the detections of every clip into same-class tracks over consecutive frames.

    Args:
        predictions: Table with `PREDICTION_COLUMNS`.
        images: Image table with `image_id`, `clip`, and `frame_id` of the predicted frames.
        link_iou: Minimum IoU of a link between consecutive frames.
        min_score: Detections below this score are not linked and get no track.

    Returns:
        The predictions with `clip`, `frame_id`, and int64 `track_id` (null for detections
        below `min_score`), unique over the whole table.
    """
    located = predictions.select(PREDICTION_COLUMNS).join(
        images.select("image_id", "clip", "frame_id"), on="image_id"
    )
    if located.height != predictions.height:
        raise ValueError("Every predicted image_id needs a clip and frame_id in `images`")
    linkable = located.filter(pl.col("score") >= min_score).sort("clip", "class_id", "frame_id")
    parts, offset = [], 0
    for _, group in linkable.partition_by("clip", "class_id", as_dict=True).items():
        track_ids = link_track_ids(
            group["frame_id"].to_numpy(),
            group.select(BOX_COLUMNS).to_numpy().astype(np.float64),
            link_iou,
        )
        parts.append(group.with_columns(track_id=pl.Series(track_ids + offset, dtype=pl.Int64)))
        offset += int(track_ids.max()) + 1
    unlinked = located.filter(pl.col("score") < min_score)
    return pl.concat(
        [*parts, unlinked.with_columns(track_id=pl.lit(None, dtype=pl.Int64))],
        how="vertical",
    )


def rescore_tracks(
    tracked: pl.DataFrame, weight: float, statistic: str, is_causal: bool
) -> pl.DataFrame:
    """Move every tracked detection score toward a statistic of its track's scores.

    Seq-NMS rescores linked detections with the average score of their sequence
    (https://arxiv.org/abs/1602.08465); here a detection's new score is
    `(1 - weight) * score + weight * track_score`, where `track_score` is the mean or the
    maximum score along its track. A track of one detection keeps its score, so detections
    that appear in a single frame are ranked below equally scored ones confirmed by the next
    frames. Detections without a track keep their score. A causal `track_score` uses only the
    current and earlier frames of the track (running mean or maximum), as an online detector
    would; otherwise it uses the whole track.

    Args:
        tracked: Result of `assign_tracks`, possibly with extra columns.
        weight: Share of the track score in the new score, in [0, 1].
        statistic: `mean` or `max` of the scores along a track.
        is_causal: Whether the track statistic covers only the frames up to the current one.

    Returns:
        The columns of `tracked` with the new `score`, sorted by `track_id` and `frame_id`.
    """
    if statistic not in TRACK_STATISTICS:
        raise ValueError(f"statistic must be one of {TRACK_STATISTICS}, got {statistic!r}")
    if not 0 <= weight <= 1:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    score = pl.col("score")
    if not is_causal:
        track_score = getattr(score, statistic)().over("track_id")
    elif statistic == "max":
        track_score = score.cum_max().over("track_id")
    else:
        track_score = (score.cum_sum() / score.cum_count()).over("track_id")
    return tracked.sort("track_id", "frame_id").with_columns(
        pl.when(pl.col("track_id").is_null())
        .then(score)
        .otherwise((1 - weight) * score + weight * track_score)
        .alias("score")
    )


def rescore_along_tracks(
    predictions: pl.DataFrame,
    images: pl.DataFrame,
    link_iou: float,
    min_score: float,
    weight: float,
    statistic: str,
    is_causal: bool,
) -> pl.DataFrame:
    """Link the detections into tracks and rescore them along the tracks with `rescore_tracks`.

    Args:
        predictions: Table with `PREDICTION_COLUMNS` on consecutive frames of clips.
        images: Image table with `image_id`, `clip`, and `frame_id` of the predicted frames.
        link_iou: Minimum IoU of a link between consecutive frames.
        min_score: Detections below this score are neither linked nor rescored.
        weight: Share of the track score in the new score, in [0, 1].
        statistic: `mean` or `max` of the scores along a track.
        is_causal: Whether the track statistic covers only the frames up to the current one.

    Returns:
        The same detections with `PREDICTION_COLUMNS` and the new `score`, sorted by
        `image_id` and descending score.
    """
    tracked = assign_tracks(predictions, images, link_iou, min_score)
    return (
        rescore_tracks(tracked, weight, statistic, is_causal)
        .select(PREDICTION_COLUMNS)
        .sort("image_id", "score", descending=[False, True], maintain_order=True)
    )


def smooth_boxes_along_tracks(
    predictions: pl.DataFrame,
    images: pl.DataFrame,
    link_iou: float,
    min_score: float,
    is_causal: bool,
) -> pl.DataFrame:
    """Average every box with the boxes of its track in the neighbouring frames.

    The box of a detection in the middle of its track from `assign_tracks` becomes
    `(previous + 2 * current + next) / 4`, coordinate by coordinate, a binomial filter that
    reduces frame-to-frame jitter and leaves constant motion unchanged. The causal filter
    drops the next frame: every box with a previous box in its track becomes
    `(previous + 2 * current) / 3`, which lags behind moving objects. Other detections keep
    their boxes; scores are unchanged.

    Args:
        predictions: Table with `PREDICTION_COLUMNS` on consecutive frames of clips.
        images: Image table with `image_id`, `clip`, and `frame_id` of the predicted frames.
        link_iou: Minimum IoU of a link between consecutive frames.
        min_score: Detections below this score are neither linked nor smoothed.
        is_causal: Whether the filter uses only the previous frame instead of both neighbours.

    Returns:
        The same detections with `PREDICTION_COLUMNS` and the smoothed boxes, sorted by
        `image_id` and descending score.
    """
    tracked = assign_tracks(predictions, images, link_iou, min_score).sort("track_id", "frame_id")
    has_previous = pl.col("frame_id").shift(1).over("track_id").is_not_null()
    has_next = pl.col("frame_id").shift(-1).over("track_id").is_not_null()
    is_smoothed = pl.col("track_id").is_not_null() & has_previous & (is_causal | has_next)
    neighbours = [1] if is_causal else [1, -1]
    smoothed = tracked.with_columns(
        pl.when(is_smoothed)
        .then(
            pl.sum_horizontal(
                2 * pl.col(column),
                *(pl.col(column).shift(shift).over("track_id") for shift in neighbours),
            )
            / (2 + len(neighbours))
        )
        .otherwise(pl.col(column))
        .alias(column)
        for column in BOX_COLUMNS
    )
    return smoothed.select(PREDICTION_COLUMNS).sort(
        "image_id", "score", descending=[False, True], maintain_order=True
    )
