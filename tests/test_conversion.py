import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.annotations import BOX_COLUMNS, AnnotationTables

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
    rows = np.loadtxt(label_path, ndmin=2)
    class_id, x_center, y_center, w, h = rows.T
    return np.stack(
        [
            class_id,
            (x_center - w / 2) * width,
            (y_center - h / 2) * height,
            w * width,
            h * height,
        ],
        axis=1,
    )


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
