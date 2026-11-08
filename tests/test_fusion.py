import numpy as np
import polars as pl
import pytest

from src.evaluation.coco_metrics import PREDICTION_COLUMNS
from src.inference.fusion import (
    cluster_weighted_boxes,
    fuse_predictions,
    fuse_weighted_boxes,
    fuse_with_nms,
)


def build_predictions(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "image_id": pl.Int64,
            "class_id": pl.Int64,
            "score": pl.Float64,
            "x": pl.Float64,
            "y": pl.Float64,
            "w": pl.Float64,
            "h": pl.Float64,
        },
        orient="row",
    )


def test_identical_members_fuse_into_the_same_boxes_and_scores() -> None:
    member = build_predictions([(1, 0, 0.9, 10, 10, 20, 20), (1, 2, 0.4, 50, 50, 10, 10)])
    fused = fuse_weighted_boxes([member, member], [1.0, 1.0], iou_thresh=0.55)
    assert fused.columns == list(PREDICTION_COLUMNS)
    assert fused.sort("class_id").rows() == member.rows()


def test_a_box_found_by_one_of_two_members_keeps_half_its_score() -> None:
    first = build_predictions([(1, 0, 0.8, 10, 10, 20, 20)])
    second = build_predictions([(1, 0, 0.6, 100, 100, 20, 20)])
    fused = fuse_weighted_boxes([first, second], [1.0, 1.0], iou_thresh=0.55)
    assert fused["score"].to_list() == pytest.approx([0.4, 0.3])


def test_overlapping_boxes_average_their_corners_weighted_by_score() -> None:
    first = build_predictions([(1, 0, 0.75, 10, 10, 20, 20)])
    second = build_predictions([(1, 0, 0.25, 12, 10, 20, 20)])
    [row] = fuse_weighted_boxes([first, second], [1.0, 1.0], iou_thresh=0.55).rows()
    assert row == pytest.approx((1, 0, 0.5, 10.5, 10, 20, 20))


def test_classes_and_frames_never_fuse_with_each_other() -> None:
    first = build_predictions([(1, 0, 0.9, 10, 10, 20, 20), (2, 0, 0.9, 10, 10, 20, 20)])
    second = build_predictions([(1, 1, 0.9, 10, 10, 20, 20)])
    fused = fuse_weighted_boxes([first, second], [1.0, 1.0], iou_thresh=0.55)
    assert sorted(fused.select("image_id", "class_id").rows()) == [(1, 0), (1, 1), (2, 0)]
    assert fused["score"].to_list() == pytest.approx([0.45] * 3)


def test_member_weights_shift_scores_and_corners_and_keep_scores_in_range() -> None:
    first = build_predictions([(1, 0, 1.0, 10, 10, 20, 20), (1, 0, 1.0, 11, 10, 20, 20)])
    second = build_predictions([(1, 0, 0.5, 14, 10, 20, 20)])
    [row] = fuse_weighted_boxes([first, second], [2.0, 1.0], iou_thresh=0.5).rows()
    assert row[2] == pytest.approx(4.5 / 5)
    assert row[3] == pytest.approx((2 * 10 + 2 * 11 + 0.5 * 14) / 4.5)
    assert 0 <= row[2] <= 1


def test_clusters_join_the_best_overlapping_fused_box_above_the_threshold() -> None:
    xyxy = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [1, 0, 11, 10], [5, 0, 15, 10]], float)
    fused, score_sums, weight_sums = cluster_weighted_boxes(
        xyxy, np.array([0.9, 0.8, 0.7, 0.6]), np.ones(4), iou_thresh=0.6
    )
    assert len(fused) == 3
    assert score_sums.tolist() == pytest.approx([1.6, 0.8, 0.6])
    assert weight_sums.tolist() == [2, 1, 1]


def test_fusion_keeps_the_500_highest_scores_per_frame() -> None:
    rows = [(1, 0, (i + 1) / 1000, 30.0 * i, 0, 10, 10) for i in range(600)]
    fused = fuse_weighted_boxes([build_predictions(rows)], [1.0], iou_thresh=0.55)
    assert fused.height == 500
    assert fused["score"].min() == pytest.approx(0.101)


def test_nms_fusion_keeps_the_best_weighted_box_of_each_overlap() -> None:
    first = build_predictions([(1, 0, 0.9, 10, 10, 20, 20), (1, 1, 0.3, 10, 10, 20, 20)])
    second = build_predictions([(1, 0, 0.8, 11, 10, 20, 20), (1, 0, 0.7, 60, 60, 5, 5)])
    fused = fuse_with_nms([first, second], [1.0, 0.5], iou_thresh=0.5)
    assert fused.rows() == [
        (1, 0, 0.9, 10, 10, 20, 20),
        (1, 0, 0.35, 60, 60, 5, 5),
        (1, 1, 0.3, 10, 10, 20, 20),
    ]


def test_unknown_fusion_methods_and_missing_weights_are_rejected() -> None:
    member = build_predictions([(1, 0, 0.9, 10, 10, 20, 20)])
    with pytest.raises(ValueError, match="Fusion method must be one of"):
        fuse_predictions([member], [1.0], "mean", 0.5)
    with pytest.raises(ValueError, match="one weight per member"):
        fuse_predictions([member, member], [1.0], "wbf", 0.5)
