import numpy as np
import polars as pl
import pytest

from src.evaluation.coco_metrics import PREDICTION_COLUMNS
from src.inference.temporal import (
    assign_tracks,
    link_track_ids,
    match_greedily,
    rescore_along_tracks,
    rescore_tracks,
    smooth_boxes_along_tracks,
)
from tests.test_fusion import build_predictions

IMAGES = pl.DataFrame(
    {
        "image_id": [1, 2, 3, 4, 5, 6],
        "clip": ["a", "a", "a", "a", "b", "b"],
        "frame_id": [1, 2, 3, 5, 1, 2],
    }
)


def test_greedy_matching_is_one_to_one_and_takes_the_highest_iou_first() -> None:
    ious = np.array([[0.9, 0.8], [0.85, 0.2], [0.1, 0.4]])
    rows, columns = match_greedily(ious, link_iou=0.3)
    assert sorted(zip(rows.tolist(), columns.tolist())) == [(0, 0), (2, 1)]
    rows, columns = match_greedily(ious, link_iou=0.95)
    assert rows.size == columns.size == 0


def test_tracks_continue_over_consecutive_frames_and_end_at_gaps() -> None:
    frame_ids = np.array([1, 1, 2, 3, 5])
    boxes = np.array([[0, 0, 10, 10], [50, 50, 10, 10], [1, 0, 10, 10], [2, 0, 10, 10]], float)
    boxes = np.concatenate([boxes, [[2, 0, 10, 10]]])
    track_ids = link_track_ids(frame_ids, boxes, link_iou=0.5)
    assert track_ids[0] == track_ids[2] == track_ids[3]
    assert len({track_ids[0], track_ids[1], track_ids[4]}) == 3


def test_tracks_never_join_classes_or_clips() -> None:
    predictions = build_predictions(
        [
            (1, 0, 0.9, 0, 0, 10, 10),
            (2, 0, 0.5, 0, 0, 10, 10),
            (2, 1, 0.5, 0, 0, 10, 10),
            (5, 0, 0.5, 0, 0, 10, 10),
            (6, 0, 0.5, 0, 0, 10, 10),
            (3, 0, 0.0001, 0, 0, 10, 10),
        ]
    )
    tracked = assign_tracks(predictions, IMAGES, link_iou=0.5, min_score=0.001)
    track_of = {(row["image_id"], row["class_id"]): row["track_id"] for row in tracked.to_dicts()}
    assert track_of[(1, 0)] == track_of[(2, 0)]
    assert track_of[(5, 0)] == track_of[(6, 0)] != track_of[(1, 0)]
    assert track_of[(2, 1)] not in (track_of[(1, 0)], track_of[(5, 0)])
    assert track_of[(3, 0)] is None


def test_scores_move_toward_the_track_statistic_and_singletons_keep_theirs() -> None:
    predictions = build_predictions(
        [
            (1, 0, 0.9, 0, 0, 10, 10),
            (2, 0, 0.3, 1, 0, 10, 10),
            (3, 0, 0.6, 2, 0, 10, 10),
            (4, 0, 0.7, 2, 0, 10, 10),
            (5, 2, 0.05, 0, 0, 10, 10),
        ]
    )
    rescored = rescore_along_tracks(
        predictions, IMAGES, 0.5, 0.1, weight=0.5, statistic="mean", is_causal=False
    )
    assert rescored.columns == list(PREDICTION_COLUMNS)
    assert rescored.sort("image_id")["score"].to_list() == pytest.approx(
        [0.75, 0.45, 0.6, 0.7, 0.05]
    )
    maximum = rescore_along_tracks(
        predictions, IMAGES, 0.5, 0.1, weight=1.0, statistic="max", is_causal=False
    )
    assert maximum.sort("image_id")["score"].to_list()[:3] == pytest.approx([0.9] * 3)
    unchanged = rescore_along_tracks(
        predictions, IMAGES, 0.5, 0.0, weight=0.0, statistic="mean", is_causal=False
    )
    assert unchanged.sort("image_id").equals(predictions.sort("image_id"))


@pytest.mark.parametrize(
    ("statistic", "expected"), [("max", [0.3, 0.9, 0.75]), ("mean", [0.3, 0.75, 0.6])]
)
def test_causal_rescoring_uses_only_the_current_and_earlier_frames(
    statistic: str, expected: list[float]
) -> None:
    predictions = build_predictions(
        [(1, 0, 0.3, 0, 0, 10, 10), (2, 0, 0.9, 1, 0, 10, 10), (3, 0, 0.6, 2, 0, 10, 10)]
    )
    rescored = rescore_along_tracks(
        predictions, IMAGES, 0.5, 0.0, weight=0.5, statistic=statistic, is_causal=True
    )
    assert rescored.sort("image_id")["score"].to_list() == pytest.approx(expected)


def test_rescoring_tracked_detections_keeps_their_extra_columns() -> None:
    predictions = build_predictions([(1, 0, 0.2, 0, 0, 10, 10), (2, 0, 0.8, 0, 0, 10, 10)])
    tracked = assign_tracks(predictions, IMAGES, 0.5, 0.0).with_columns(raw_score=pl.col("score"))
    rescored = rescore_tracks(tracked, weight=1.0, statistic="max", is_causal=False)
    assert rescored["raw_score"].to_list() == pytest.approx([0.2, 0.8])
    assert rescored["score"].to_list() == pytest.approx([0.8, 0.8])


def test_causal_smoothing_uses_only_the_previous_box() -> None:
    predictions = build_predictions(
        [
            (1, 0, 0.9, 10, 0, 10, 10),
            (2, 0, 0.8, 13, 0, 10, 10),
            (3, 0, 0.7, 16, 0, 13, 10),
        ]
    )
    smoothed = smooth_boxes_along_tracks(
        predictions, IMAGES, link_iou=0.3, min_score=0.0, is_causal=True
    )
    boxes = {row[0]: row[3:] for row in smoothed.rows()}
    assert boxes[1] == pytest.approx((10.0, 0.0, 10.0, 10.0))
    assert boxes[2] == pytest.approx((12.0, 0.0, 10.0, 10.0))
    assert boxes[3] == pytest.approx((15.0, 0.0, 12.0, 10.0))


def test_boxes_inside_tracks_are_smoothed_and_constant_motion_is_kept() -> None:
    predictions = build_predictions(
        [
            (1, 0, 0.9, 10, 0, 10, 10),
            (2, 0, 0.8, 13, 0, 10, 10),
            (3, 0, 0.7, 14, 0, 12, 10),
            (1, 1, 0.9, 0, 50, 10, 10),
            (2, 1, 0.9, 2, 50, 10, 10),
            (3, 1, 0.9, 4, 50, 10, 10),
            (4, 0, 0.6, 14, 0, 12, 10),
        ]
    )
    smoothed = smooth_boxes_along_tracks(
        predictions, IMAGES, link_iou=0.3, min_score=0.0, is_causal=False
    )
    assert smoothed.columns == list(PREDICTION_COLUMNS)
    rows = {(row[0], row[1]): row[2:] for row in smoothed.rows()}
    assert rows[(2, 0)] == pytest.approx((0.8, 12.5, 0.0, 10.5, 10.0))
    assert rows[(2, 1)] == pytest.approx((0.9, 2.0, 50.0, 10.0, 10.0))
    for key in ((1, 0), (3, 0), (4, 0), (1, 1), (3, 1)):
        assert (
            rows[key]
            == predictions.filter(
                (pl.col("image_id") == key[0]) & (pl.col("class_id") == key[1])
            ).row(0)[2:]
        )


def test_invalid_rescoring_settings_and_unknown_frames_are_rejected() -> None:
    predictions = build_predictions([(1, 0, 0.9, 0, 0, 10, 10)])
    with pytest.raises(ValueError, match="statistic must be one of"):
        rescore_along_tracks(predictions, IMAGES, 0.5, 0.0, 0.5, "median", False)
    with pytest.raises(ValueError, match="weight must be in"):
        rescore_along_tracks(predictions, IMAGES, 0.5, 0.0, 1.5, "mean", False)
    with pytest.raises(ValueError, match="needs a clip and frame_id"):
        assign_tracks(build_predictions([(9, 0, 0.9, 0, 0, 10, 10)]), IMAGES, 0.5, 0.0)
