import json
from pathlib import Path

import polars as pl
import pytest
from faster_coco_eval.core.cocoeval import Params

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.classes import CLASS_NAMES
from src.evaluation.coco_metrics import (
    SIZE_RANGES,
    build_coco_ground_truth,
    evaluate_detections,
    validate_predictions,
)

CAR = CLASS_NAMES.index("car")
VAN = CLASS_NAMES.index("van")
PERSON = CLASS_NAMES.index("person")


def build_tables(
    objects: list[tuple[int, int, int, int, int, int]],
    ignored: tuple[tuple[int, int, int, int, int], ...] = (),
    frame_sizes: tuple[tuple[int, int], ...] = ((400, 400),),
) -> AnnotationTables:
    images = pl.DataFrame(
        [("val", i, w, h) for i, (w, h) in enumerate(frame_sizes, start=1)],
        schema=["split", "image_id", "width", "height"],
        orient="row",
    )
    object_table = pl.DataFrame(
        [("val", *row) for row in objects],
        schema=["split", "image_id", "class_id", *BOX_COLUMNS],
        orient="row",
    )
    ignored_table = pl.DataFrame(
        [("val", *row) for row in ignored],
        schema={"split": pl.String, **{c: pl.Int64 for c in ["image_id", *BOX_COLUMNS]}},
        orient="row",
    )
    return AnnotationTables(images=images, objects=object_table, ignored=ignored_table)


def to_predictions(objects: pl.DataFrame, score: float = 0.9) -> pl.DataFrame:
    return objects.select(
        "image_id", "class_id", pl.lit(score).alias("score"), pl.col(BOX_COLUMNS).cast(pl.Float64)
    )


def empty_predictions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"image_id": pl.Int64, "class_id": pl.Int64, "score": pl.Float64}
        | {column: pl.Float64 for column in BOX_COLUMNS}
    )


MIXED_OBJECTS = [
    (1, CAR, 10, 10, 40, 30),
    (1, CAR, 30, 15, 40, 30),
    (1, PERSON, 200, 200, 8, 20),
    (2, VAN, 100, 100, 120, 110),
    (2, PERSON, 5, 5, 6, 14),
]


def test_predictions_equal_to_ground_truth_give_map_one() -> None:
    tables = build_tables(MIXED_OBJECTS, frame_sizes=((1920, 1080), (3840, 2160)))
    metrics = evaluate_detections(tables, "val", to_predictions(tables.objects))
    assert metrics["map"] == metrics["map50"] == metrics["map75"] == 1.0
    assert metrics["ap_small"] == metrics["ap_medium"] == metrics["ap_large"] == 1.0
    assert metrics["map_max_width_1920"] == metrics["ap_small_max_width_1920"] == 1.0
    assert (metrics["frames"], metrics["objects"], metrics["frames_max_width_1920"]) == (2, 5, 1)
    present = {CLASS_NAMES[c] for c in (CAR, VAN, PERSON)}
    for name, average_precision in metrics["per_class_ap"].items():
        assert average_precision == (1.0 if name in present else None)


def test_empty_predictions_give_map_zero() -> None:
    tables = build_tables(MIXED_OBJECTS)
    metrics = evaluate_detections(tables, "val", empty_predictions())
    assert metrics["map"] == metrics["map50"] == metrics["ap_small"] == 0.0
    assert metrics["per_class_ap"]["car"] == 0.0


def test_false_positive_inside_ignored_region_does_not_lower_map() -> None:
    tables = build_tables(MIXED_OBJECTS[:3], ignored=((1, 300, 300, 60, 60),))
    predictions = to_predictions(tables.objects, score=0.5)
    inside = pl.DataFrame(
        [(1, CAR, 0.99, 310.0, 310.0, 20.0, 20.0), (1, VAN, 0.98, 310.0, 330.0, 40.0, 20.0)],
        schema=predictions.columns,
        orient="row",
    )
    outside = inside.with_columns(pl.col("y") - 250)
    assert evaluate_detections(tables, "val", pl.concat([predictions, inside]))["map"] == 1.0
    assert evaluate_detections(tables, "val", pl.concat([predictions, outside]))["map"] < 1.0


def test_class_ids_shift_to_category_ids_exactly_once() -> None:
    tables = build_tables([(1, CAR, 10, 10, 40, 30)])
    metrics = evaluate_detections(tables, "val", to_predictions(tables.objects))
    assert metrics["per_class_ap"] == {
        name: (1.0 if name == "car" else None) for name in CLASS_NAMES
    }
    shifted = to_predictions(tables.objects).with_columns(pl.col("class_id") + 1)
    assert evaluate_detections(tables, "val", shifted)["map"] == 0.0


def test_coco_ground_truth_matches_provided_annotations(
    data_root: Path, smoke_annotations: AnnotationTables
) -> None:
    for (split,), images in smoke_annotations.images.group_by("split"):
        keys = images.select("split", "image_id")
        tables = AnnotationTables(
            images=images,
            objects=smoke_annotations.objects.join(keys, on=["split", "image_id"]),
            ignored=smoke_annotations.ignored.join(keys, on=["split", "image_id"]),
        )
        built = build_coco_ground_truth(tables)
        source_split = images["source_split"][0]
        coco = json.loads((data_root / source_split / "annotations_coco.json").read_text())
        source_ids = dict(images.select("source_image_id", "image_id").iter_rows())
        expected = [
            (source_ids[a["image_id"]], a["category_id"], a["iscrowd"], *map(float, a["bbox"]))
            for a in coco["annotations"]
            if a["image_id"] in source_ids
        ]
        loaded = [
            (a["image_id"], a["category_id"], a["iscrowd"], *a["bbox"])
            for a in built["annotations"]
        ]
        assert expected, split
        assert sorted(a for a in loaded if not a[2]) == sorted(a for a in expected if not a[2])
        assert {a for a in loaded if a[2]} == {a for a in expected if a[2]}


def test_smoke_ground_truth_as_predictions_gives_map_one(
    smoke_annotations: AnnotationTables,
) -> None:
    val_objects = smoke_annotations.objects.filter(pl.col("split") == "val")
    metrics = evaluate_detections(smoke_annotations, "val", to_predictions(val_objects))
    assert metrics["map"] == 1.0
    assert metrics["objects"] == val_objects.height


def test_size_ranges_are_the_standard_coco_ranges() -> None:
    params = Params(iouType="bbox")
    assert dict(zip(params.areaRngLbl, params.areaRng)) == {"all": [0, 1e5**2], **SIZE_RANGES}


def test_boxes_on_bucket_boundaries_count_in_both_buckets() -> None:
    tables = build_tables(
        [
            (1, CAR, 0, 0, 31, 33),
            (1, CAR, 50, 0, 32, 32),
            (1, CAR, 100, 0, 95, 97),
            (1, CAR, 200, 200, 96, 96),
        ]
    )
    detected = tables.objects.filter(pl.col("w") == pl.col("h"))
    metrics = evaluate_detections(tables, "val", to_predictions(detected))
    assert metrics["ap_small"] == pytest.approx(51 / 101)
    assert metrics["ap_medium"] == pytest.approx(67 / 101)
    assert metrics["ap_large"] == 1.0


def test_metrics_on_frames_at_most_1920_px_wide() -> None:
    tables = build_tables(
        [(1, CAR, 10, 10, 20, 20), (2, CAR, 10, 10, 20, 20)],
        frame_sizes=((1920, 1080), (1921, 1080)),
    )
    predictions = to_predictions(tables.objects.filter(pl.col("image_id") == 1))
    metrics = evaluate_detections(tables, "val", predictions)
    assert metrics["frames_max_width_1920"] == 1
    assert metrics["map_max_width_1920"] == metrics["ap_small_max_width_1920"] == 1.0
    assert metrics["map"] < 1.0


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (pl.col("image_id") + 5, "unknown image_id"),
        (pl.lit(8).alias("class_id"), "class_id outside 0-7"),
        (pl.lit(1.5).alias("score"), "score outside"),
        (pl.lit(0.0).alias("w"), "non-positive w or h"),
        (pl.lit(float("nan")).alias("x"), "non-finite value"),
        (pl.lit(None, pl.Float64).alias("score"), "null value"),
    ],
)
def test_invalid_predictions_are_rejected(change: pl.Expr, problem: str) -> None:
    tables = build_tables(MIXED_OBJECTS)
    predictions = to_predictions(tables.objects).with_columns(change)
    with pytest.raises(ValueError, match=problem):
        validate_predictions(predictions, tables.images)


def test_more_than_500_predictions_per_frame_are_rejected() -> None:
    tables = build_tables(MIXED_OBJECTS[:1])
    predictions = pl.concat([to_predictions(tables.objects)] * 501)
    with pytest.raises(ValueError, match="more than 500"):
        validate_predictions(predictions, tables.images)
    validate_predictions(predictions.head(500), tables.images)
