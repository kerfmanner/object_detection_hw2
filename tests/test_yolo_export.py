import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from torchvision.io import decode_image
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths
from ultralytics.utils import YAML

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.boxes import convert_xywh_to_yolo
from src.data.classes import CLASS_NAMES
from src.data.dataset import load_frame_steps, select_frames
from src.data.yolo_export import (
    clip_boxes_to_tile,
    compute_tile_offsets,
    locate_yolo_label,
    write_yolo_dataset,
    write_yolo_tile_dataset,
)
from tests.conftest import (
    DATA_CONFIG_PATH,
    SYNTHETIC_IGNORED_BOX,
    SYNTHETIC_SOURCE_CLIPS,
    SYNTHETIC_SPLIT_CLIPS,
)
from tests.test_splits import SWAPPED_INTO_TRAIN, SWAPPED_INTO_VAL


def read_list(data_yaml: dict, split: str) -> list[str]:
    return Path(data_yaml[split]).read_text().splitlines()


def find_source_split(clip: str) -> str:
    return next(source for source, clips in SYNTHETIC_SOURCE_CLIPS.items() if clip in clips)


def snapshot_files(root: Path) -> set[tuple[str, int, int]]:
    return {
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }


def load_with_ultralytics(data_yaml: dict) -> dict[str, YOLODataset]:
    return {
        split: YOLODataset(
            img_path=data_yaml[split], data=data_yaml | {"channels": 3}, augment=False, imgsz=96
        )
        for split in ("train", "val")
    }


def test_synthetic_export_lists_symlinks_to_the_selected_frames(
    synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    export_dir = tmp_path / "export"
    data_yaml_path = write_yolo_dataset(
        synthetic_annotations.images, {"train": 5, "val": 1}, export_dir
    )
    data_yaml = YAML.load(data_yaml_path)
    assert data_yaml_path == export_dir.resolve() / "data.yaml"
    assert data_yaml["path"] == str(export_dir.resolve())
    assert data_yaml["nc"] == len(CLASS_NAMES)
    assert data_yaml["names"] == list(CLASS_NAMES)
    listed_clips = {}
    for split, frame_step in (("train", 5), ("val", 1)):
        listed = [Path(path) for path in read_list(data_yaml, split)]
        expected = select_frames(
            synthetic_annotations.images.filter(pl.col("split") == split), frame_step
        )
        assert [path.readlink() for path in listed] == [
            Path(path) for path in expected["file_path"]
        ]
        for path in listed:
            assert path.is_symlink()
            assert path.parents[2] == export_dir.resolve() and path.parents[1].name == "images"
        listed_clips[split] = sorted({path.parent.name for path in listed})
    assert listed_clips == {split: sorted(clips) for split, clips in SYNTHETIC_SPLIT_CLIPS.items()}


def test_synthetic_label_copies_come_from_each_clip_source_split(
    synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    data_yaml = YAML.load(
        write_yolo_dataset(synthetic_annotations.images, {"train": 1, "val": 1}, tmp_path / "out")
    )
    for split in ("train", "val"):
        listed: list[str | Path] = [*read_list(data_yaml, split)]
        for image_link, label_copy in zip(listed, img2label_paths(listed), strict=True):
            source_image = Path(image_link).readlink()
            source_label = locate_yolo_label(str(source_image))
            assert source_label.parents[2].name == find_source_split(source_image.parent.name)
            assert not Path(label_copy).is_symlink()
            assert Path(label_copy).read_bytes() == source_label.read_bytes()


def test_ultralytics_reads_exported_labels_without_writing_to_data_root(
    synthetic_root: Path, synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    before = snapshot_files(synthetic_root)
    export_dir = tmp_path / "export"
    images = synthetic_annotations.images
    data_yaml = YAML.load(write_yolo_dataset(images, {"train": 1, "val": 1}, export_dir))
    for split, dataset in load_with_ultralytics(data_yaml).items():
        split_images = images.filter(pl.col("split") == split)
        assert len(dataset.labels) == split_images.height
        for label in dataset.labels:
            image = split_images.row(
                by_predicate=pl.col("file_path") == str(Path(label["im_file"]).readlink()),
                named=True,
            )
            objects = synthetic_annotations.objects.filter(
                pl.col("split") == split, pl.col("image_id") == image["image_id"]
            )
            boxes = objects.select(BOX_COLUMNS).to_numpy().astype(np.float64).reshape(-1, 4)
            expected = convert_xywh_to_yolo(boxes, image["width"], image["height"])
            assert label["cls"].ravel().tolist() == objects["class_id"].to_list()
            assert np.abs(label["bboxes"] - expected).max(initial=0) < 1e-4
    assert snapshot_files(synthetic_root) == before
    assert len(list((export_dir / "labels").glob("*.cache"))) == 2


def test_export_refuses_an_existing_directory(
    synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    with pytest.raises(FileExistsError):
        write_yolo_dataset(synthetic_annotations.images, {"train": 1}, tmp_path)


def test_export_rejects_frames_without_label_files_before_writing(
    synthetic_root: Path, synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(synthetic_root, data_root)
    (data_root / "val" / "labels" / "clip_moved_to_train" / "0000002.txt").unlink()
    images = synthetic_annotations.images.with_columns(
        pl.col("file_path").str.replace(str(synthetic_root), str(data_root), literal=True)
    )
    with pytest.raises(ValueError, match="1 frames of 'train' without labels"):
        write_yolo_dataset(images, {"val": 1, "train": 1}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_adopted_split_export_copies_swapped_clip_labels_from_source_split(
    data_root: Path, annotations: AnnotationTables, tmp_path: Path
) -> None:
    frame_steps = load_frame_steps(DATA_CONFIG_PATH)
    split_steps = {"train": frame_steps["train_tuning"], "val": frame_steps["val_monitoring"]}
    data_yaml = YAML.load(write_yolo_dataset(annotations.images, split_steps, tmp_path / "out"))
    listed = {split: read_list(data_yaml, split) for split in split_steps}
    assert {split: len(paths) for split, paths in listed.items()} == {"train": 4919, "val": 520}
    for split, clip, source_split in (
        ("train", SWAPPED_INTO_TRAIN, "val"),
        ("val", SWAPPED_INTO_VAL, "train"),
    ):
        clip_paths: list[str | Path] = [
            path for path in listed[split] if Path(path).parent.name == clip
        ]
        assert clip_paths
        for image_link, label_copy in zip(clip_paths, img2label_paths(clip_paths), strict=True):
            source_image = Path(image_link).readlink()
            assert source_image.parents[2] == data_root / source_split
            source_label = data_root / source_split / "labels" / clip / Path(label_copy).name
            assert Path(label_copy).read_bytes() == source_label.read_bytes()


def test_ultralytics_on_adopted_split_writes_nothing_into_data_root(
    data_root: Path, annotations: AnnotationTables, smoke_config: dict, tmp_path: Path
) -> None:
    clips = [smoke_config["train_clip"], smoke_config["val_clip"], SWAPPED_INTO_TRAIN]
    clips.append(SWAPPED_INTO_VAL)
    images = annotations.images.filter(
        pl.col("clip").is_in(clips),
        pl.col("frame_id").is_in(smoke_config["frame_ids"]),
    )
    before = snapshot_files(data_root)
    export_dir = tmp_path / "export"
    data_yaml = YAML.load(write_yolo_dataset(images, {"train": 1, "val": 1}, export_dir))
    datasets = load_with_ultralytics(data_yaml)
    assert {split: len(dataset.labels) for split, dataset in datasets.items()} == {
        "train": 16,
        "val": 16,
    }
    assert snapshot_files(data_root) == before
    assert len(list((export_dir / "labels").glob("*.cache"))) == 2


SYNTHETIC_FILL_VALUE = 200
SYNTHETIC_TILES = {
    "frame_width": 64,
    "tile_size": 40,
    "min_overlap": 16,
    "min_visible_fraction": 0.25,
    "fill": [SYNTHETIC_FILL_VALUE] * 3,
}


def test_tile_offsets_cover_the_side_with_the_minimum_overlap() -> None:
    assert compute_tile_offsets(1920, 1088, 256) == [0, 832]
    assert compute_tile_offsets(1080, 1088, 256) == [0]
    assert compute_tile_offsets(1088, 1088, 256) == [0]
    offsets = compute_tile_offsets(3000, 1088, 256)
    assert offsets[0] == 0 and offsets[-1] == 3000 - 1088
    assert all(1088 - (b - a) >= 256 for a, b in zip(offsets, offsets[1:]))


def test_boxes_are_clipped_to_the_tile_and_slivers_dropped() -> None:
    boxes = np.array([[10.0, 10.0, 20.0, 30.0], [35.0, 0.0, 45.0, 10.0], [39.0, 0.0, 59.0, 10.0]])
    kept, tile_boxes = clip_boxes_to_tile(boxes, (0, 0, 40, 40), 0.25)
    assert kept.tolist() == [0, 1]
    assert tile_boxes.tolist() == [[10.0, 10.0, 10.0, 20.0], [35.0, 0.0, 5.0, 10.0]]
    kept, tile_boxes = clip_boxes_to_tile(boxes, (24, 0, 64, 40), 0.25)
    assert kept.tolist() == [1, 2]
    assert tile_boxes.tolist() == [[11.0, 0.0, 10.0, 10.0], [15.0, 0.0, 20.0, 10.0]]
    kept, _ = clip_boxes_to_tile(boxes, (24, 8, 64, 48), 0.25)
    assert kept.tolist() == []


def test_synthetic_tiles_hold_the_resized_frames_and_clipped_labels(
    synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    export_dir = tmp_path / "tiles"
    data_yaml = YAML.load(
        write_yolo_tile_dataset(
            synthetic_annotations, {"train": 1, "val": 1}, export_dir, SYNTHETIC_TILES, {"train"}
        )
    )
    images = synthetic_annotations.images
    datasets = load_with_ultralytics(data_yaml)
    for split, dataset in datasets.items():
        expected_tiles = 0
        for frame in images.filter(pl.col("split") == split).iter_rows(named=True):
            height = round(frame["height"] * SYNTHETIC_TILES["frame_width"] / frame["width"])
            expected_tiles += len(compute_tile_offsets(height, 40, 16)) * 2
        assert len(dataset.labels) == expected_tiles
    clip_a = images.filter(pl.col("clip") == "clip_a", pl.col("frame_id") == 1).row(0, named=True)
    first_tile = export_dir / "images" / "clip_a" / "0000001_0.jpg"
    assert decode_image(str(first_tile)).shape == (3, 40, 40)
    objects = synthetic_annotations.objects.filter(
        pl.col("split") == "train", pl.col("image_id") == clip_a["image_id"]
    )
    xywh = objects.select(BOX_COLUMNS).to_numpy().astype(np.float64)
    is_inside = (xywh[:, 0] + xywh[:, 2] <= 40) & (xywh[:, 1] + xywh[:, 3] <= 40)
    assert objects["class_id"].to_numpy()[is_inside].tolist() == [0, 1, 2, 3, 4]
    assert xywh[5].tolist() == [33.0, 22.0, 9.0, 8.0] and xywh[6, 0] == 39.0
    expected = np.concatenate([xywh[is_inside], [[33.0, 22.0, 7.0, 8.0]]])
    rows = np.loadtxt(export_dir / "labels" / "clip_a" / "0000001_0.txt", ndmin=2)
    assert rows[:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.abs(rows[:, 1:] - convert_xywh_to_yolo(expected, 40, 40)).max() < 1e-5


def test_tiles_fill_ignored_regions_only_in_masked_splits(
    synthetic_annotations: AnnotationTables, tmp_path: Path
) -> None:
    write_yolo_tile_dataset(
        synthetic_annotations, {"train": 1}, tmp_path / "masked", SYNTHETIC_TILES, {"train"}
    )
    write_yolo_tile_dataset(
        synthetic_annotations, {"train": 1}, tmp_path / "plain", SYNTHETIC_TILES, set()
    )
    x, y, w, _ = SYNTHETIC_IGNORED_BOX
    tile_x, tile_y = x + w - 2 - 24, y + 4
    frame_value, fill_value = 40, SYNTHETIC_FILL_VALUE
    for name, is_filled in (("masked", True), ("plain", False)):
        tile = decode_image(str(tmp_path / name / "images" / "clip_a" / "0000001_1.jpg"))
        pixel = tile[:, tile_y, tile_x].float().mean().item()
        assert (abs(pixel - fill_value) < abs(pixel - frame_value)) == is_filled
