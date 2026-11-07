import numpy as np
import polars as pl
import pytest

from src.evaluation.failure_analysis import (
    compute_proposal_recall,
    compute_recall_by_size,
    count_error_categories,
    find_best_f1_threshold,
    match_detections,
    select_error_examples,
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
    assert matches.detections["best_class_id"].to_list() == [
        CAR,
        CAR,
        PERSON,
        CAR,
        None,
        None,
        None,
        None,
        CAR,
        CAR,
        CAR,
    ]


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
        ("all", 6, 3),
        ("small", 4, 2),
        ("medium", 1, 1),
        ("large", 1, 0),
    ]
    assert recall["recall"].to_list() == [0.5, 0.5, 1.0, 0.0]


def test_recall_by_size_counts_boundary_objects_in_both_buckets() -> None:
    tables = build_tables([(1, CAR, 0, 0, 32, 32), (1, CAR, 100, 0, 96, 96)])
    detections = detections_table([(1, CAR, 0.9, 0, 0, 32, 32)])
    recall = compute_recall_by_size(match_detections(tables, "val", detections))
    assert recall.select("size_bucket", "objects", "matched").rows() == [
        ("all", 2, 1),
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


def test_error_examples_take_the_most_confident_detections_and_random_misses_per_clip() -> None:
    tables = build_tables(OBJECTS, ignored=IGNORED, frame_sizes=((400, 400), (400, 400)))
    matches = match_detections(tables, "val", detections_table(DETECTIONS))
    images = pl.DataFrame({"image_id": [1, 2], "clip": ["a", "b"]})
    examples = select_error_examples(matches, images, threshold=0.5, per_category=2, seed=42)
    assert examples.columns == ["category", "image_id", "class_id", "score", "x", "y", "w", "h"]
    rows = examples.filter(pl.col("category") != "miss").select("category", "image_id", "score")
    assert rows.rows() == [
        ("correct", 1, 0.95),
        ("correct", 2, 0.5),
        ("duplicate", 1, 0.9),
        ("wrong_class", 1, 0.85),
        ("poor_localisation", 1, 0.8),
        ("background", 1, 0.75),
    ]
    misses = examples.filter(pl.col("category") == "miss")
    assert misses["image_id"].sort().to_list() == [1, 2] and misses["score"].is_null().all()
    reordered = matches._replace(objects=matches.objects.reverse())
    again = select_error_examples(reordered, images, threshold=0.5, per_category=2, seed=42)
    assert again.equals(examples)
    one_clip = images.with_columns(pl.lit("a").alias("clip"))
    single = select_error_examples(matches, one_clip, threshold=0.5, per_category=2, seed=42)
    assert single.filter(pl.col("category") == "correct")["score"].to_list() == [0.95]
    assert single.filter(pl.col("category") == "miss").height == 1


def test_proposal_recall_counts_objects_covered_by_the_top_proposals() -> None:
    objects = pl.DataFrame(
        {
            "image_id": [1, 1, 2],
            "x": [0, 100, 0],
            "y": [0, 0, 0],
            "w": [10, 100, 10],
            "h": [10, 100, 10],
        }
    )
    proposals = pl.DataFrame(
        {
            "image_id": [1, 1, 1],
            "rank": [1, 0, 2],
            "x": [100.0, 50.0, 0.0],
            "y": [0.0, 50.0, 0.0],
            "w": [100.0, 10.0, 12.0],
            "h": [80.0, 10.0, 10.0],
        }
    )
    recall = compute_proposal_recall(
        objects, proposals, top_ns=[1, 2, 3], iou_thresholds=[0.5, 0.9]
    )
    rows = recall.filter(pl.col("size_bucket") == "all").select("iou_threshold", "top_n", "recall")
    assert rows.rows() == [
        (0.5, 1, 0.0),
        (0.5, 2, 1 / 3),
        (0.5, 3, 2 / 3),
        (0.9, 1, 0.0),
        (0.9, 2, 0.0),
        (0.9, 3, 0.0),
    ]
    small = recall.filter(pl.col("size_bucket") == "small", pl.col("iou_threshold") == 0.5)
    assert small.select("top_n", "objects", "recall").rows() == [
        (1, 2, 0.0),
        (2, 2, 0.0),
        (3, 2, 0.5),
    ]
    assert recall["size_bucket"].unique(maintain_order=True).to_list() == ["all", "small", "large"]
