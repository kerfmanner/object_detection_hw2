import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from decouple import config
from torchvision.io import write_jpeg

from src.data.annotations import AnnotationTables, load_annotations, load_split_clips
from src.data.boxes import convert_xywh_to_yolo
from src.data.classes import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "data.toml"


@pytest.fixture(scope="session")
def data_root() -> Path:
    return Path(config("DATA_ROOT"))


@pytest.fixture(scope="session")
def smoke_config() -> dict:
    return tomllib.loads(DATA_CONFIG_PATH.read_text())["smoke"]


@pytest.fixture(scope="session")
def split_clips() -> dict[str, list[str]]:
    return load_split_clips(DATA_CONFIG_PATH)


@pytest.fixture(scope="session")
def annotations(data_root: Path, split_clips: dict[str, list[str]]) -> AnnotationTables:
    return load_annotations(data_root, split_clips)


@pytest.fixture(scope="session")
def smoke_annotations(annotations: AnnotationTables, smoke_config: dict) -> AnnotationTables:
    smoke_images = annotations.images.filter(
        pl.col("clip").is_in([smoke_config["train_clip"], smoke_config["val_clip"]]),
        pl.col("frame_id").is_in(smoke_config["frame_ids"]),
    )
    keys = smoke_images.select("split", "image_id")
    return AnnotationTables(
        images=smoke_images,
        objects=annotations.objects.join(keys, on=["split", "image_id"], maintain_order="left"),
        ignored=annotations.ignored.join(keys, on=["split", "image_id"], maintain_order="left"),
    )


SYNTHETIC_SOURCE_CLIPS = {
    "train": {"clip_a": (64, 48, 6), "clip_moved_to_val": (80, 40, 3)},
    "val": {"clip_moved_to_train": (72, 56, 3)},
}
SYNTHETIC_SPLIT_CLIPS = {"train": ["clip_a", "clip_moved_to_train"], "val": ["clip_moved_to_val"]}
SYNTHETIC_EMPTY_FRAME = ("clip_moved_to_train", 3)
SYNTHETIC_IGNORED_BOX = [50, 30, 10, 8]


def build_synthetic_frame_boxes(frame_id: int) -> np.ndarray:
    class_ids = np.arange(len(CLASS_NAMES))
    return np.stack(
        [2 + 6 * class_ids + frame_id, 2 + 4 * class_ids, 4 + class_ids, 3 + class_ids], axis=1
    )


def write_synthetic_source_split(source_dir: Path, clips: dict[str, tuple[int, int, int]]) -> None:
    images, annotations = [], []
    for clip, (width, height, frame_count) in clips.items():
        for frame_id in range(1, frame_count + 1):
            image_id = len(images) + 1
            file_name = f"images/{clip}/{frame_id:07d}.jpg"
            images.append(
                {
                    "id": image_id,
                    "video": clip,
                    "frame_id": frame_id,
                    "width": width,
                    "height": height,
                    "file_name": file_name,
                }
            )
            image_path = source_dir / file_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pixels = torch.full((3, height, width), 40 * frame_id, dtype=torch.uint8)
            write_jpeg(pixels, str(image_path))
            label_path = source_dir / "labels" / clip / f"{frame_id:07d}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if (clip, frame_id) == SYNTHETIC_EMPTY_FRAME:
                label_path.write_text("")
                continue
            boxes = build_synthetic_frame_boxes(frame_id)
            yolo_boxes = convert_xywh_to_yolo(boxes.astype(np.float64), width, height)
            label_path.write_text(
                "".join(
                    f"{class_id} {' '.join(f'{v:.6f}' for v in box)}\n"
                    for class_id, box in enumerate(yolo_boxes)
                )
            )
            for class_id, box in enumerate(boxes.tolist()):
                annotations.append(
                    {
                        "image_id": image_id,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "category_id": class_id + 1,
                        "iscrowd": 0,
                        "truncation": 0,
                        "occlusion": 0,
                        "track_id": class_id,
                        "orig_category_id": class_id + 1,
                    }
                )
            for category_id in range(1, len(CLASS_NAMES) + 1):
                annotations.append(
                    {
                        "image_id": image_id,
                        "bbox": SYNTHETIC_IGNORED_BOX,
                        "area": SYNTHETIC_IGNORED_BOX[2] * SYNTHETIC_IGNORED_BOX[3],
                        "category_id": category_id,
                        "iscrowd": 1,
                        "truncation": -1,
                        "occlusion": -1,
                        "track_id": -1,
                        "orig_category_id": 0,
                    }
                )
    categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, start=1)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (source_dir / "annotations_coco.json").write_text(json.dumps(coco))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_data")
    for source_split, clips in SYNTHETIC_SOURCE_CLIPS.items():
        write_synthetic_source_split(root / source_split, clips)
    return root


@pytest.fixture(scope="session")
def synthetic_annotations(synthetic_root: Path) -> AnnotationTables:
    return load_annotations(synthetic_root, SYNTHETIC_SPLIT_CLIPS)


@pytest.fixture
def prediction_data(
    synthetic_root: Path, synthetic_annotations: AnnotationTables, tmp_path: Path
) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    shutil.copytree(synthetic_root, data_root)
    (data_root / "test_images.csv").write_text(
        "image_id,file_name,clip,frame,width,height\n"
        "clip_001_0000001,test/images/clip_001/0000001.jpg,clip_001,1,64,48\n"
        "clip_001_0000002,test/images/clip_001/0000002.jpg,clip_001,2,64,48\n"
    )
    config_path = tmp_path / "data.toml"
    splits = "\n".join(f"{split} = {json.dumps(c)}" for split, c in SYNTHETIC_SPLIT_CLIPS.items())
    config_path.write_text(f"[splits]\n{splits}\n")
    return data_root, config_path
