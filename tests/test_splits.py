import tomllib
from pathlib import Path

import polars as pl
import pytest

from src.data.annotations import (
    AnnotationTables,
    assign_clip_folds,
    assign_splits,
    load_annotations,
    load_clip_split,
)
from tests.conftest import DATA_CONFIG_PATH

SWAPPED_INTO_TRAIN = "uav0000268_05773_v"
SWAPPED_INTO_VAL = "uav0000263_03289_v"


def test_split_config_lists_disjoint_clips_with_the_decided_swap(
    split_clips: dict[str, list[str]],
) -> None:
    assert (len(split_clips["train"]), len(split_clips["val"])) == (56, 7)
    assert not set(split_clips["train"]) & set(split_clips["val"])
    assert SWAPPED_INTO_TRAIN in split_clips["train"]
    assert SWAPPED_INTO_VAL in split_clips["val"]


def test_loaded_split_matches_config_and_covers_every_clip_once(
    annotations: AnnotationTables, split_clips: dict[str, list[str]]
) -> None:
    clip_splits = annotations.images.select("clip", "split").unique()
    assert clip_splits["clip"].is_unique().all()
    for split, clips in split_clips.items():
        assert sorted(clip_splits.filter(pl.col("split") == split)["clip"]) == sorted(clips)
    assert annotations.images.select("split", "image_id").is_unique().all()
    frames = annotations.images.group_by("split").len().sort("split")["len"].to_list()
    assert frames == [24454, 2593]


def test_assign_splits_is_deterministic_and_rejects_invalid_configs() -> None:
    images = pl.DataFrame(
        {"clip": ["b", "a", "a", "c"], "frame_id": [1, 2, 1, 1], "width": [1, 1, 1, 1]}
    )
    split_clips = {"train": ["a", "c"], "val": ["b"]}
    first = assign_splits(images, split_clips)
    assert first.equals(assign_splits(images.reverse(), split_clips))
    assert first.select("split", "image_id", "clip", "frame_id").rows() == [
        ("train", 1, "a", 1),
        ("train", 2, "a", 2),
        ("train", 3, "c", 1),
        ("val", 1, "b", 1),
    ]
    with pytest.raises(ValueError, match="unassigned"):
        assign_splits(images, {"train": ["a"], "val": ["b"]})
    with pytest.raises(ValueError, match="duplicated"):
        assign_splits(images, {"train": ["a", "b", "c"], "val": ["b"]})


def test_stored_folds_are_the_seeded_group_k_fold_of_all_clips(
    annotations: AnnotationTables, split_clips: dict[str, list[str]]
) -> None:
    folds = tomllib.loads(DATA_CONFIG_PATH.read_text())["folds"]
    assert list(folds) == ["fold_0", "fold_1", "fold_2"]
    assert list(folds.values()) == assign_clip_folds(annotations.images, 3, 42)
    assert assign_clip_folds(annotations.images.reverse(), 3, 42) == list(folds.values())
    all_clips = sorted(split_clips["train"] + split_clips["val"])
    assert sorted(clip for clips in folds.values() for clip in clips) == all_clips
    assert [len(clips) for clips in folds.values()] == [21, 21, 21]


@pytest.mark.parametrize("fold", ["fold_0", "fold_1", "fold_2"])
def test_a_fold_split_validates_on_its_fold_and_trains_on_the_others(
    fold: str, data_root: Path, split_clips: dict[str, list[str]]
) -> None:
    folds = tomllib.loads(DATA_CONFIG_PATH.read_text())["folds"]
    fold_split = load_clip_split(DATA_CONFIG_PATH, fold)
    assert fold_split["val"] == folds[fold]
    assert not set(fold_split["train"]) & set(fold_split["val"])
    assert len(fold_split["train"]) == 42
    tables = load_annotations(data_root, fold_split)
    val_clips = tables.images.filter(pl.col("split") == "val")["clip"].unique().sort()
    assert val_clips.to_list() == sorted(folds[fold])
    assert load_clip_split(DATA_CONFIG_PATH, "main") == split_clips


@pytest.mark.parametrize("clip_split", ["fold_3", "folds", "smoke", "val"])
def test_unknown_clip_splits_are_rejected(clip_split: str) -> None:
    with pytest.raises(ValueError, match="Unknown clip_split"):
        load_clip_split(DATA_CONFIG_PATH, clip_split)
