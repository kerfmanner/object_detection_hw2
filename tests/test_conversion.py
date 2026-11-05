import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.boxes import convert_xywh_to_xyxy, convert_xywh_to_yolo, convert_yolo_to_xywh
from src.data.classes import CLASS_NAMES
from tests.conftest import (
    SYNTHETIC_EMPTY_FRAME,
    SYNTHETIC_IGNORED_BOX,
    SYNTHETIC_SPLIT_CLIPS,
    build_synthetic_frame_boxes,
)

VISDRONE_IGNORED_CATEGORY_IDS = (0, 11)


@pytest.fixture(scope="module")
def raw_smoke_annotations(
    data_root: Path, smoke_annotations: AnnotationTables
) -> dict[tuple[str, int], list[dict]]:
    raw: dict[tuple[str, int], list[dict]] = {}
    for (source_split,), source_images in smoke_annotations.images.group_by("source_split"):
        coco = json.loads((data_root / str(source_split) / "annotations_coco.json").read_text())
        for image_id in source_images["source_image_id"].to_list():
            raw[(str(source_split), image_id)] = [
                a for a in coco["annotations"] if a["image_id"] == image_id
            ]
    return raw


def select_frame_objects(tables: AnnotationTables, image: dict) -> pl.DataFrame:
    return tables.objects.filter(
        pl.col("split") == image["split"], pl.col("image_id") == image["image_id"]
    )


def read_yolo_boxes_in_pixels(label_path: Path, width: int, height: int) -> np.ndarray:
    lines = label_path.read_text().splitlines()
    rows = np.array([line.split() for line in lines], dtype=np.float64).reshape(-1, 5)
    return np.concatenate([rows[:, :1], convert_yolo_to_xywh(rows[:, 1:], width, height)], axis=1)


def test_pixel_yolo_round_trip_is_within_1e_4() -> None:
    rng = np.random.default_rng(42)
    width, height = 3840, 2160
    xy = rng.uniform(0, [width - 50, height - 50], size=(1000, 2))
    wh = rng.uniform(1, 50, size=(1000, 2))
    boxes = np.concatenate([xy, wh], axis=1)
    yolo = convert_xywh_to_yolo(boxes, width, height)
    assert ((yolo >= 0) & (yolo <= 1)).all()
    assert np.abs(convert_yolo_to_xywh(yolo, width, height) - boxes).max() < 1e-4


def test_xywh_to_xyxy_adds_size_to_corner() -> None:
    boxes = np.array([[10.0, 20.0, 5.0, 7.0], [0.0, 0.0, 1.0, 1.0]])
    expected = np.array([[10.0, 20.0, 15.0, 27.0], [0.0, 0.0, 1.0, 1.0]])
    assert np.array_equal(convert_xywh_to_xyxy(boxes), expected)


def test_smoke_subset_is_not_empty(smoke_annotations: AnnotationTables) -> None:
    assert smoke_annotations.images.height == 16
    assert smoke_annotations.objects.height > 0


def test_class_id_is_category_id_minus_one(
    raw_smoke_annotations: dict[tuple[str, int], list[dict]],
    smoke_annotations: AnnotationTables,
) -> None:
    for image in smoke_annotations.images.iter_rows(named=True):
        raw = raw_smoke_annotations[(image["source_split"], image["source_image_id"])]
        expected_class_ids = [a["category_id"] - 1 for a in raw if a["iscrowd"] == 0]
        loaded_class_ids = select_frame_objects(smoke_annotations, image)["class_id"].to_list()
        assert loaded_class_ids == expected_class_ids
        assert all(0 <= class_id <= 7 for class_id in loaded_class_ids)


def test_loader_boxes_match_yolo_labels_within_one_pixel(
    smoke_annotations: AnnotationTables,
) -> None:
    for image in smoke_annotations.images.iter_rows(named=True):
        image_path = Path(image["file_path"])
        label_path = image_path.parents[2] / "labels" / image["clip"] / f"{image_path.stem}.txt"
        yolo = read_yolo_boxes_in_pixels(label_path, image["width"], image["height"])
        loaded = select_frame_objects(smoke_annotations, image).select("class_id", *BOX_COLUMNS)
        assert loaded.height == len(yolo)
        assert np.array_equal(loaded["class_id"].to_numpy(), yolo[:, 0])
        assert np.abs(loaded.select(BOX_COLUMNS).to_numpy() - yolo[:, 1:]).max() <= 1


def test_ignored_regions_never_appear_in_objects(
    raw_smoke_annotations: dict[tuple[str, int], list[dict]],
    smoke_annotations: AnnotationTables,
) -> None:
    objects = smoke_annotations.objects
    assert not objects["orig_category_id"].is_in(VISDRONE_IGNORED_CATEGORY_IDS).any()
    assert smoke_annotations.ignored.height > 0
    assert objects.join(
        smoke_annotations.ignored, on=["split", "image_id", *BOX_COLUMNS]
    ).is_empty()
    for image in smoke_annotations.images.iter_rows(named=True):
        raw = raw_smoke_annotations[(image["source_split"], image["source_image_id"])]
        crowd_count = sum(a["iscrowd"] for a in raw)
        assert select_frame_objects(smoke_annotations, image).height == len(raw) - crowd_count


def test_synthetic_loader_keeps_all_classes_and_drops_ignored_regions(
    synthetic_annotations: AnnotationTables,
) -> None:
    images, objects, ignored = synthetic_annotations
    for split, clips in SYNTHETIC_SPLIT_CLIPS.items():
        assert sorted(images.filter(pl.col("split") == split)["clip"].unique()) == sorted(clips)
    for image in images.iter_rows(named=True):
        frame_objects = select_frame_objects(synthetic_annotations, image)
        frame_ignored = ignored.filter(
            pl.col("split") == image["split"], pl.col("image_id") == image["image_id"]
        )
        if (image["clip"], image["frame_id"]) == SYNTHETIC_EMPTY_FRAME:
            assert frame_objects.is_empty() and frame_ignored.is_empty()
            continue
        assert frame_objects["class_id"].to_list() == list(range(len(CLASS_NAMES)))
        expected_boxes = build_synthetic_frame_boxes(image["frame_id"])
        assert np.array_equal(frame_objects.select(BOX_COLUMNS).to_numpy(), expected_boxes)
        assert frame_ignored.select(BOX_COLUMNS).rows() == [tuple(SYNTHETIC_IGNORED_BOX)]
    assert objects.join(ignored, on=["split", "image_id", *BOX_COLUMNS]).is_empty()


def test_synthetic_yolo_labels_match_loader_within_1e_4(
    synthetic_annotations: AnnotationTables,
) -> None:
    for image in synthetic_annotations.images.iter_rows(named=True):
        image_path = Path(image["file_path"])
        label_path = image_path.parents[2] / "labels" / image["clip"] / f"{image_path.stem}.txt"
        yolo = read_yolo_boxes_in_pixels(label_path, image["width"], image["height"])
        loaded = select_frame_objects(synthetic_annotations, image)
        assert np.array_equal(loaded["class_id"].to_numpy(), yolo[:, 0])
        assert np.abs(loaded.select(BOX_COLUMNS).to_numpy() - yolo[:, 1:]).max(initial=0) < 1e-4
