import numpy as np
import polars as pl
import pytest

from src.evaluation.failure_analysis import (
    compute_recall_by_size,
    count_error_categories,
    find_best_f1_threshold,
    match_detections,
)
from tests.test_evaluation import CAR, PERSON, VAN, build_tables, empty_predictions

OBJECTS = [
    (1, CAR, 0, 0, 10, 10),
    (1, CAR, 100, 0, 10, 10),
    (1, PERSON, 200, 0, 10, 10),
    (1, CAR, 320, 20, 10, 10),
    (2, CAR, 0, 0, 40, 40),
    (2, VAN, 100, 100, 100, 100),
]
IGNORED = ((1, 300, 0, 50, 50),)
DETECTIONS = [
    (1, CAR, 0.95, 0, 0, 10, 10, "correct", 1.0),
    (1, CAR, 0.90, 0, 0, 10, 10, "duplicate", 1.0),
    (1, VAN, 0.85, 200, 0, 10, 10, "wrong_class", 1.0),
    (1, CAR, 0.80, 100, 0, 10, 40, "poor_localisation", 0.25),
    (1, CAR, 0.75, 500, 500, 10, 10, "background", 0.0),
    (1, CAR, 0.70, 310, 30, 10, 10, "ignored", 0.0),
    (1, CAR, 0.65, 345, 0, 10, 10, "ignored", 0.0),
    (1, CAR, 0.60, 346, 0, 10, 10, "background", 0.0),
    (1, CAR, 0.55, 320, 20, 10, 10, "correct", 1.0),
    (2, CAR, 0.50, 0, 0, 40, 40, "correct", 1.0),
    (2, PERSON, 0.45, 0, 0, 20, 20, "poor_localisation", 0.25),
]
PREDICTION_COLUMNS = ["image_id", "class_id", "score", "x", "y", "w", "h"]


def detections_table(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        [row[:7] for row in rows], schema=PREDICTION_COLUMNS, orient="row"
    ).with_columns(pl.col("x", "y", "w", "h").cast(pl.Float64))


def test_each_detection_gets_the_defined_category() -> None:
    tables = build_tables(OBJECTS, ignored=IGNORED, frame_sizes=((400, 400), (400, 400)))
    shuffled = detections_table(DETECTIONS).sample(fraction=1.0, shuffle=True, seed=42)
    matches = match_detections(tables, "val", shuffled)
    assert matches.detections["category"].to_list() == [row[7] for row in DETECTIONS]
    assert np.allclose(matches.detections["best_iou"].to_numpy(), [row[8] for row in DETECTIONS])
    assert matches.objects["matched_score"].to_list() == [0.95, None, None, 0.55, 0.5, None]


def test_matching_follows_score_order_not_iou() -> None:
    tables = build_tables([(1, CAR, 0, 0, 10, 10)])
    detections = detections_table([(1, CAR, 0.5, 0, 0, 10, 10), (1, CAR, 0.9, 0, 0, 10, 12)])
    matches = match_detections(tables, "val", detections)
    assert matches.detections.select("score", "category").rows() == [
        (0.9, "correct"),
        (0.5, "duplicate"),
    ]


def test_frames_without_objects_or_detections() -> None:
    tables = build_tables([(1, CAR, 0, 0, 10, 10)], frame_sizes=((400, 400), (400, 400)))
    detections = detections_table([(2, CAR, 0.9, 0, 0, 10, 10)])
    matches = match_detections(tables, "val", detections)
    assert matches.detections["category"].to_list() == ["background"]
    assert matches.objects["matched_score"].to_list() == [None]
    empty = match_detections(tables, "val", empty_predictions())
    assert empty.detections.is_empty()
    assert find_best_f1_threshold(empty) == {
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert count_error_categories(empty, 0.5).row(-1) == ("miss", 1)


def test_recall_by_size_counts_matched_objects_per_bucket() -> None:
    tables = build_tables(OBJECTS, ignored=IGNORED, frame_sizes=((400, 400), (400, 400)))
    matches = match_detections(tables, "val", detections_table(DETECTIONS))
    recall = compute_recall_by_size(matches)
    assert recall.select("size_bucket", "objects", "matched").rows() == [
        ("small", 4, 2),
        ("medium", 1, 1),
        ("large", 1, 0),
    ]
    assert recall["recall"].to_list() == [0.5, 1.0, 0.0]


def test_recall_by_size_counts_boundary_objects_in_both_buckets() -> None:
    tables = build_tables([(1, CAR, 0, 0, 32, 32), (1, CAR, 100, 0, 96, 96)])
    detections = detections_table([(1, CAR, 0.9, 0, 0, 32, 32)])
    recall = compute_recall_by_size(match_detections(tables, "val", detections))
    assert recall.select("size_bucket", "objects", "matched").rows() == [
        ("small", 1, 1),
        ("medium", 2, 1),
        ("large", 1, 0),
    ]


def test_best_f1_threshold_ignores_detections_in_ignored_regions() -> None:
    tables = build_tables(
        [(1, CAR, 0, 0, 10, 10), (1, CAR, 100, 0, 10, 10)], ignored=((1, 300, 0, 50, 50),)
    )
    detections = detections_table(
        [
            (1, CAR, 0.99, 310, 10, 10, 10),
            (1, CAR, 0.9, 0, 0, 10, 10),
            (1, CAR, 0.8, 500, 0, 10, 10),
            (1, CAR, 0.7, 100, 0, 10, 10),
            (1, CAR, 0.7, 600, 0, 10, 10),
            (1, CAR, 0.6, 700, 0, 10, 10),
        ]
    )
    matches = match_detections(tables, "val", detections)
    best = find_best_f1_threshold(matches)
    assert best["threshold"] == 0.9
    assert best["f1"] == pytest.approx(2 / 3)
    assert best["precision"] == 1.0 and best["recall"] == 0.5
    errors = count_error_categories(matches, 0.7)
    assert dict(errors.iter_rows()) == {
        "correct": 2,
        "duplicate": 0,
        "wrong_class": 0,
        "poor_localisation": 0,
        "background": 2,
        "miss": 0,
    }
    assert count_error_categories(matches, 0.9).row(-1) == ("miss", 1)


def test_error_breakdown_counts_every_category() -> None:
    tables = build_tables(OBJECTS, ignored=IGNORED, frame_sizes=((400, 400), (400, 400)))
    matches = match_detections(tables, "val", detections_table(DETECTIONS))
    errors = dict(count_error_categories(matches, 0.0).iter_rows())
    assert errors == {
        "correct": 3,
        "duplicate": 1,
        "wrong_class": 1,
        "poor_localisation": 2,
        "background": 2,
        "miss": 3,
    }
