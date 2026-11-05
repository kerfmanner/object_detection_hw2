import numpy as np
import polars as pl
import pytest

from src.data.boxes import (
    add_ignored_coverage_column,
    add_max_iou_columns,
    compute_coverage_matrix,
    compute_covered_fraction,
    compute_iou_matrix,
    compute_max_iou_with_others,
    compute_paired_iou,
    explode_size_buckets,
    is_in_size_bucket,
)


def test_size_buckets_include_both_ends_as_in_coco() -> None:
    areas = pl.DataFrame({"area": [0, 1023, 1024, 1025, 9215, 9216, 9217]})
    flags = areas.select(
        is_in_size_bucket(pl.col("area"), bucket).alias(bucket)
        for bucket in ("small", "medium", "large")
    )
    assert flags.rows() == [
        (True, False, False),
        (True, False, False),
        (True, True, False),
        (False, True, False),
        (False, True, False),
        (False, True, True),
        (False, False, True),
    ]


def test_explode_size_buckets_repeats_boundary_rows() -> None:
    boxes = pl.DataFrame(
        {"name": ["a", "b", "c", "d"], "w": [10, 32, 50, 96], "h": [10, 32, 50, 96]}
    )
    exploded = explode_size_buckets(boxes, pl.col("w") * pl.col("h"))
    assert exploded.select("size_bucket", "name").cast({"size_bucket": pl.String}).rows() == [
        ("small", "a"),
        ("small", "b"),
        ("medium", "b"),
        ("medium", "c"),
        ("medium", "d"),
        ("large", "d"),
    ]
    assert exploded.columns == ["name", "w", "h", "size_bucket"]


def test_coverage_matrix_divides_intersection_by_box_area() -> None:
    boxes = np.array([[0, 0, 10, 10], [5, 0, 10, 10]], dtype=float)
    regions = np.array([[0, 0, 5, 10], [100, 100, 5, 5]], dtype=float)
    assert np.allclose(compute_coverage_matrix(boxes, regions), [[0.5, 0.0], [0.0, 0.0]])
    assert compute_coverage_matrix(boxes, np.zeros((0, 4))).shape == (2, 0)


def test_compute_iou_matrix_handles_identical_disjoint_touching_and_partial_boxes() -> None:
    boxes_a = np.array([[0, 0, 10, 10]], dtype=float)
    boxes_b = np.array(
        [[0, 0, 10, 10], [20, 20, 5, 5], [10, 0, 10, 10], [5, 0, 10, 10]], dtype=float
    )
    iou = compute_iou_matrix(boxes_a, boxes_b)
    assert iou.shape == (1, 4)
    assert iou[0] == pytest.approx([1.0, 0.0, 0.0, 50 / 150])


def test_max_iou_with_others_ignores_self_and_other_groups() -> None:
    boxes = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [100, 100, 4, 4]], dtype=float)
    assert compute_max_iou_with_others(boxes) == pytest.approx([1 / 3, 1 / 3, 0.0])
    grouped = compute_max_iou_with_others(boxes, np.array([0, 1, 1]))
    assert grouped == pytest.approx([0.0, 0.0, 0.0])
    assert compute_max_iou_with_others(boxes[:1]) == pytest.approx([0.0])


def test_add_max_iou_columns_compares_only_within_a_frame() -> None:
    objects = pl.DataFrame(
        {
            "split": ["train", "train", "val"],
            "image_id": [1, 1, 1],
            "class_id": [0, 2, 0],
            "x": [0, 5, 0],
            "y": [0, 0, 0],
            "w": [10, 10, 10],
            "h": [10, 10, 10],
        }
    )
    result = add_max_iou_columns(objects)
    assert result["max_iou_any_class"].to_list() == pytest.approx([1 / 3, 1 / 3, 0.0])
    assert result["max_iou_same_class"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_compute_covered_fraction_counts_overlapping_regions_once() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 4, 4], [2, 2, 2, 2]], dtype=float)
    half = np.array([[0, 0, 5, 10]], dtype=float)
    assert compute_covered_fraction(boxes[:1], half) == pytest.approx([0.5])
    overlapping = np.array([[0, 0, 6, 10], [4, 0, 4, 10]], dtype=float)
    fractions = compute_covered_fraction(boxes, overlapping)
    assert fractions == pytest.approx([0.8, 0.8, 0.0, 1.0])
    assert compute_covered_fraction(boxes, np.zeros((0, 4))) == pytest.approx([0, 0, 0, 0])


def test_add_ignored_coverage_column_uses_regions_of_the_same_frame() -> None:
    boxes = pl.DataFrame(
        {
            "split": ["train", "train", "val"],
            "image_id": [1, 2, 1],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "w": [10, 10, 10],
            "h": [10, 10, 10],
        }
    )
    ignored = pl.DataFrame(
        {"split": ["train"], "image_id": [1], "x": [0], "y": [0], "w": [10], "h": [5]}
    )
    result = add_ignored_coverage_column(boxes, ignored).sort("split", "image_id")
    assert result["ignored_fraction"].to_list() == pytest.approx([0.5, 0.0, 0.0])


def test_compute_paired_iou_matches_iou_matrix_diagonal() -> None:
    boxes_a = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 2, 2]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [10, 10, 2, 2]], dtype=float)
    expected = np.diag(compute_iou_matrix(boxes_a, boxes_b))
    assert compute_paired_iou(boxes_a, boxes_b) == pytest.approx(expected)
    assert compute_paired_iou(boxes_a, boxes_b) == pytest.approx([1.0, 1 / 3, 0.0])
