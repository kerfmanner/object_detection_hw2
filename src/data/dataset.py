import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.augment import FrameTransform
from src.data.boxes import convert_xywh_to_xyxy


def load_frame_steps(data_config_path: Path) -> dict[str, int]:
    """Read the `[frame_steps]` table of the data configuration.

    Args:
        data_config_path: Path of `config/data.toml`.

    Returns:
        Mapping from step name (e.g. `train_tuning`, `val_reporting`) to a frame step of at
        least 1.
    """
    frame_steps = tomllib.loads(data_config_path.read_text())["frame_steps"]
    invalid = {
        name: step for name, step in frame_steps.items() if type(step) is not int or step < 1
    }
    if invalid:
        raise ValueError(f"Frame steps must be integers of at least 1: {invalid}")
    return frame_steps


def select_frames(images: pl.DataFrame, frame_step: int) -> pl.DataFrame:
    """Keep the first frame of every clip and every `frame_step`-th frame after it.

    Args:
        images: Image table with `split`, `clip`, and `frame_id`.
        frame_step: Distance between kept frames; 1 keeps all frames.

    Returns:
        The rows of the kept frames in their original order.
    """
    first_frame = pl.col("frame_id").min().over("split", "clip")
    return images.filter((pl.col("frame_id") - first_frame) % frame_step == 0)


def select_smoke_frames(images: pl.DataFrame, data_config_path: Path) -> pl.DataFrame:
    """Keep the frames of the smoke subset defined in the `[smoke]` table of the data config.

    Args:
        images: Image table with `clip` and `frame_id`.
        data_config_path: Path of `config/data.toml`.

    Returns:
        The rows of the configured frames of the smoke train and val clips, in their order.
    """
    smoke = tomllib.loads(data_config_path.read_text())["smoke"]
    return images.filter(
        pl.col("clip").is_in([smoke["train_clip"], smoke["val_clip"]]),
        pl.col("frame_id").is_in(smoke["frame_ids"]),
    )


def index_frame_rows(table: pl.DataFrame, image_ids: np.ndarray) -> np.ndarray:
    """Locate the rows of each frame in a table sorted by `image_id`.

    Args:
        table: Rows of one split sorted by `image_id`.
        image_ids: Array of shape (n,) with the image ids to locate.

    Returns:
        Array of shape (n, 2) with the start and end row of each frame.
    """
    row_image_ids = table["image_id"].to_numpy()
    return np.stack(
        [
            np.searchsorted(row_image_ids, image_ids, side="left"),
            np.searchsorted(row_image_ids, image_ids, side="right"),
        ],
        axis=1,
    )


def select_xyxy_boxes(table: pl.DataFrame) -> torch.Tensor:
    """Convert the pixel box columns of a table to a corner-coordinate tensor.

    Args:
        table: Table with pixel box columns `x`, `y`, `w`, `h`.

    Returns:
        Tensor of shape (n, 4) with `x1, y1, x2, y2` in px as float32.
    """
    boxes = table.select(BOX_COLUMNS).to_numpy().astype(np.float32).reshape(-1, 4)
    return torch.from_numpy(convert_xywh_to_xyxy(boxes))


class VisDroneDataset(Dataset):
    """Frames of one split, read from disk on access, with their labelled objects.

    Annotations are converted to tensors once; `__getitem__` decodes a single frame and
    copies its annotation rows, so DataLoader workers never send the shared tensors.
    Each item is a pair of a uint8 image tensor of shape (3, height, width) in RGB order and
    a target dictionary with:
        `image_id`: int64 scalar, the frame's `image_id` within the split;
        `boxes`: float32 tensor (n, 4) with `x1, y1, x2, y2` in px of the labelled objects;
        `labels`: int64 tensor (n,) with 0-based class ids;
        `ignored_boxes`: float32 tensor (k, 4) with `x1, y1, x2, y2` in px of the ignored
            regions, which are never part of `boxes`.

    Args:
        tables: Annotation tables from `load_annotations`.
        split: Configured split name, `train` or `val`.
        frame_step: Frame step from `load_frame_steps`; 1 keeps all frames.
        transform: Optional transform of each decoded frame and its target, e.g. from
            `build_train_transform`; None returns them unchanged.

    Attributes:
        images: Image table of the selected frames, ordered by `image_id`.
    """

    def __init__(
        self,
        tables: AnnotationTables,
        split: str,
        frame_step: int,
        transform: FrameTransform | None = None,
    ) -> None:
        self.images = select_frames(tables.images.filter(pl.col("split") == split), frame_step)
        self.images = self.images.sort("image_id")
        self.image_ids = self.images["image_id"].to_numpy()
        self.file_paths = self.images["file_path"].to_list()
        self.frame_sizes = self.images.select("height", "width").rows()
        objects, ignored = (
            table.filter(pl.col("split") == split).sort("image_id", maintain_order=True)
            for table in (tables.objects, tables.ignored)
        )
        self.object_bounds = index_frame_rows(objects, self.image_ids)
        self.object_boxes = select_xyxy_boxes(objects)
        self.object_labels = torch.from_numpy(objects["class_id"].to_numpy().astype(np.int64))
        self.ignored_bounds = index_frame_rows(ignored, self.image_ids)
        self.ignored_boxes = select_xyxy_boxes(ignored)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        file_path = self.file_paths[index]
        image = decode_image(file_path, mode=ImageReadMode.RGB)
        if tuple(image.shape[1:]) != self.frame_sizes[index]:
            raise ValueError(
                f"{file_path} has size {tuple(image.shape[1:])}, annotations give "
                f"{self.frame_sizes[index]} (height, width)"
            )
        object_start, object_end = self.object_bounds[index]
        ignored_start, ignored_end = self.ignored_bounds[index]
        target = {
            "image_id": torch.tensor(self.image_ids[index], dtype=torch.int64),
            "boxes": self.object_boxes[object_start:object_end].clone(),
            "labels": self.object_labels[object_start:object_end].clone(),
            "ignored_boxes": self.ignored_boxes[ignored_start:ignored_end].clone(),
        }
        if self.transform is not None:
            return self.transform(image, target)
        return image, target
