import numpy as np
import polars as pl
import pytest

from src.models.anchors import (
    cluster_box_shapes,
    compute_best_anchor_iou,
    compute_shape_iou,
    load_default_anchors,
)


def test_compute_shape_iou_ignores_position() -> None:
    iou = compute_shape_iou(np.array([[10.0, 10.0]]), np.array([[10.0, 10.0], [20.0, 20.0]]))
    assert iou[0] == pytest.approx([1.0, 0.25])


def test_cluster_box_shapes_separates_two_groups_reproducibly() -> None:
    rng = np.random.default_rng(0)
    shapes = np.vstack(
        [rng.normal([10, 20], 0.5, (200, 2)), rng.normal([60, 30], 1.0, (200, 2))]
    ).round()
    centroids, mean_iou = cluster_box_shapes(shapes, 2, seed=42, max_iterations=100)
    assert centroids == pytest.approx(np.array([[10, 20], [60, 30]]), abs=1.0)
    assert mean_iou > 0.9
    again, _ = cluster_box_shapes(shapes, 2, seed=42, max_iterations=100)
    assert np.array_equal(centroids, again)


def test_load_default_anchors_reads_faster_rcnn_levels() -> None:
    anchors = load_default_anchors("fasterrcnn_resnet50_fpn_v2")
    assert anchors.height == 15
    assert anchors["stride"].unique().sort().to_list() == [4, 8, 16, 32, 64]
    level_0 = anchors.filter(pl.col("level") == 0).select("w", "h").rows()
    assert sorted(level_0) == [(22.0, 46.0), (32.0, 32.0), (46.0, 22.0)]


def test_best_anchor_iou_is_one_on_the_grid_and_bounded_by_shape_iou() -> None:
    anchors = pl.DataFrame(
        {
            "stride": [8],
            "x1": [-8.0],
            "y1": [-8.0],
            "x2": [8.0],
            "y2": [8.0],
            "w": [16.0],
            "h": [16.0],
        }
    )
    boxes = np.array([[16.0, 16.0, 16.0, 16.0], [20.0, 20.0, 16.0, 16.0], [0.0, 0.0, 4.0, 4.0]])
    grid = compute_best_anchor_iou(boxes, anchors, is_grid_aligned=True)
    shape = compute_best_anchor_iou(boxes, anchors, is_grid_aligned=False)
    assert grid[0] == pytest.approx(1.0)
    assert grid[1] == pytest.approx((12 * 12) / (2 * 256 - 144))
    assert np.all(grid <= shape + 1e-12)
    assert shape[2] == pytest.approx(16 / 256)
