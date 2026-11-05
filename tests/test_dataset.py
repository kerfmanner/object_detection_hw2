from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.boxes import convert_xywh_to_xyxy
from src.data.classes import CLASS_NAMES
from src.data.dataset import VisDroneDataset, load_frame_steps, select_frames
from tests.conftest import (
    DATA_CONFIG_PATH,
    SYNTHETIC_EMPTY_FRAME,
    SYNTHETIC_IGNORED_BOX,
    build_synthetic_frame_boxes,
)


def select_frame_rows(table: pl.DataFrame, split: str, image_id: int) -> pl.DataFrame:
    return table.filter(pl.col("split") == split, pl.col("image_id") == image_id)


def to_xyxy_tensor(table: pl.DataFrame) -> torch.Tensor:
    boxes = table.select(BOX_COLUMNS).to_numpy().astype(np.float32).reshape(-1, 4)
    return torch.from_numpy(convert_xywh_to_xyxy(boxes))


def test_frame_steps_come_from_data_config() -> None:
    assert load_frame_steps(DATA_CONFIG_PATH) == {
        "train_tuning": 5,
        "train_final": 2,
        "val_monitoring": 5,
        "val_reporting": 1,
    }


@pytest.mark.parametrize("step", ["0", "2.5", '"5"', "true"])
def test_invalid_frame_steps_are_rejected(tmp_path: Path, step: str) -> None:
    config_path = tmp_path / "data.toml"
    config_path.write_text(f"[frame_steps]\ntrain_tuning = {step}\n")
    with pytest.raises(ValueError, match="train_tuning"):
        load_frame_steps(config_path)


def test_select_frames_keeps_every_step_from_first_frame_of_each_clip() -> None:
    images = pl.DataFrame(
        {
            "split": ["train"] * 7 + ["val"] * 3,
            "clip": ["a"] * 7 + ["b"] * 3,
            "frame_id": [1, 2, 3, 4, 5, 6, 7, 3, 4, 5],
        }
    )
    assert select_frames(images, 1).equals(images)
    assert select_frames(images, 2)["frame_id"].to_list() == [1, 3, 5, 7, 3, 5]
    assert select_frames(images, 5)["frame_id"].to_list() == [1, 6, 3]


@pytest.mark.parametrize(
    ("step", "expected"), [(1, [24454, 2593]), (2, [12240, 1299]), (5, [4919, 520])]
)
def test_select_frames_on_adopted_split_gives_documented_counts(
    annotations: AnnotationTables, step: int, expected: list[int]
) -> None:
    counts = select_frames(annotations.images, step).group_by("split").len().sort("split")
    assert counts["len"].to_list() == expected


def test_synthetic_dataset_returns_all_classes_and_separate_ignored_regions(
    synthetic_annotations: AnnotationTables,
) -> None:
    for split in ("train", "val"):
        dataset = VisDroneDataset(synthetic_annotations, split, frame_step=1)
        assert dataset.images.equals(synthetic_annotations.images.filter(pl.col("split") == split))
        for index, image in enumerate(dataset.images.iter_rows(named=True)):
            pixels, target = dataset[index]
            assert pixels.dtype == torch.uint8
            assert pixels.shape == (3, image["height"], image["width"])
            assert int(target["image_id"]) == image["image_id"]
            assert target["boxes"].dtype == torch.float32
            assert target["labels"].dtype == torch.int64
            if (image["clip"], image["frame_id"]) == SYNTHETIC_EMPTY_FRAME:
                assert target["boxes"].shape == (0, 4)
                assert target["labels"].shape == (0,)
                assert target["ignored_boxes"].shape == (0, 4)
                continue
            assert target["labels"].tolist() == list(range(len(CLASS_NAMES)))
            expected_boxes = convert_xywh_to_xyxy(build_synthetic_frame_boxes(image["frame_id"]))
            assert torch.equal(target["boxes"], torch.from_numpy(expected_boxes.astype(np.float32)))
            expected_ignored = convert_xywh_to_xyxy(np.array([SYNTHETIC_IGNORED_BOX]))
            assert target["ignored_boxes"].tolist() == expected_ignored.tolist()
            is_ignored_box = (target["boxes"][:, None] == target["ignored_boxes"][None]).all(-1)
            assert not is_ignored_box.any()


def test_synthetic_dataset_applies_frame_step_per_clip(
    synthetic_annotations: AnnotationTables,
) -> None:
    dataset = VisDroneDataset(synthetic_annotations, "train", frame_step=5)
    assert dataset.images.select("clip", "frame_id").rows() == [
        ("clip_a", 1),
        ("clip_a", 6),
        ("clip_moved_to_train", 1),
    ]
    assert [int(dataset[i][1]["image_id"]) for i in range(len(dataset))] == dataset.images[
        "image_id"
    ].to_list()


def test_items_do_not_share_storage_with_the_dataset(
    synthetic_annotations: AnnotationTables,
) -> None:
    dataset = VisDroneDataset(synthetic_annotations, "train", frame_step=1)
    _, target = dataset[1]
    for key in ("boxes", "labels", "ignored_boxes"):
        assert target[key].untyped_storage().nbytes() == target[key].nbytes


def test_dataset_rejects_frames_whose_size_differs_from_annotations(
    synthetic_annotations: AnnotationTables,
) -> None:
    images = synthetic_annotations.images.with_columns(pl.col("width") + 1)
    dataset = VisDroneDataset(synthetic_annotations._replace(images=images), "val", frame_step=1)
    with pytest.raises(ValueError, match="has size"):
        dataset[0]


@pytest.mark.parametrize("split", ["train", "val"])
def test_smoke_dataset_matches_annotation_tables(
    smoke_annotations: AnnotationTables, split: str
) -> None:
    dataset = VisDroneDataset(smoke_annotations, split, frame_step=1)
    assert len(dataset) == 8
    for index, image in enumerate(dataset.images.iter_rows(named=True)):
        pixels, target = dataset[index]
        assert pixels.shape == (3, image["height"], image["width"])
        objects = select_frame_rows(smoke_annotations.objects, split, image["image_id"])
        ignored = select_frame_rows(smoke_annotations.ignored, split, image["image_id"])
        assert torch.equal(target["boxes"], to_xyxy_tensor(objects))
        assert target["labels"].tolist() == objects["class_id"].to_list()
        assert torch.equal(target["ignored_boxes"], to_xyxy_tensor(ignored))
        assert ((target["labels"] >= 0) & (target["labels"] < len(CLASS_NAMES))).all()
