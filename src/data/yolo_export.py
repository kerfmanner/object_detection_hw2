import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torchvision.io import ImageReadMode, decode_image, write_jpeg

from src.data.annotations import BOX_COLUMNS, AnnotationTables
from src.data.augment import mask_ignored_regions
from src.data.boxes import convert_xywh_to_xyxy, convert_xywh_to_yolo
from src.data.classes import CLASS_NAMES
from src.data.dataset import select_frames

TILE_JPEG_QUALITY = 95
TILE_WRITER_THREADS = 8
FRAME_KEY = ("split", "image_id")


def locate_yolo_label(file_path: str) -> Path:
    """Locate the provided YOLO label file of a frame in the frame's source split.

    Args:
        file_path: Absolute path `<source_split>/images/<clip>/<frame>.jpg` of a frame.

    Returns:
        Path `<source_split>/labels/<clip>/<frame>.txt`.
    """
    image_path = Path(file_path)
    return image_path.parents[2] / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def write_yolo_dataset(images: pl.DataFrame, frame_steps: dict[str, int], output_dir: Path) -> Path:
    """Write an Ultralytics dataset of the configured splits into a new directory.

    Every selected frame gets a symlink `<output_dir>/images/<clip>/<frame>.jpg` to the
    provided image and a copy `<output_dir>/labels/<clip>/<frame>.txt` of its label file
    from the clip's source split, also for clips moved to another split. Ultralytics finds
    labels by replacing the last `images` directory with `labels`
    (`ultralytics.data.utils.img2label_paths`), so it reads the copies and writes its
    `.cache` files into `output_dir`, never into the data root.

    Args:
        images: Image table with `split`, `clip`, `frame_id`, and absolute `file_path`.
        frame_steps: Mapping from split name to its frame step, e.g. `{"train": 5, "val": 5}`.
        output_dir: Directory to create, e.g. under `runs/`; it must not exist yet.

    Returns:
        Path of the written `data.yaml`, which lists `<output_dir>/<split>.txt` per split.
    """
    split_source_paths = {
        split: select_frames(images.filter(pl.col("split") == split), frame_step)
        .sort("clip", "frame_id")["file_path"]
        .to_list()
        for split, frame_step in frame_steps.items()
    }
    for split, source_paths in split_source_paths.items():
        if not source_paths:
            raise ValueError(f"No frames in split {split!r}")
        missing = [path for path in source_paths if not locate_yolo_label(path).is_file()]
        if missing:
            raise ValueError(
                f"{len(missing)} frames of {split!r} without labels, e.g. {missing[0]}"
            )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True)
    split_image_paths = {}
    for split, source_paths in split_source_paths.items():
        export_paths = []
        for source_path in source_paths:
            source_image = Path(source_path)
            clip = source_image.parent.name
            image_link = output_dir / "images" / clip / source_image.name
            label_copy = output_dir / "labels" / clip / f"{source_image.stem}.txt"
            image_link.parent.mkdir(parents=True, exist_ok=True)
            label_copy.parent.mkdir(parents=True, exist_ok=True)
            image_link.symlink_to(source_image)
            shutil.copyfile(locate_yolo_label(source_path), label_copy)
            export_paths.append(str(image_link))
        split_image_paths[split] = export_paths
    return write_data_yaml(split_image_paths, output_dir)


def write_data_yaml(split_image_paths: dict[str, list[str]], output_dir: Path) -> Path:
    """Write the image list of every split and the Ultralytics `data.yaml` that lists them.

    Args:
        split_image_paths: Mapping from split name to the absolute image paths of the split.
        output_dir: Existing absolute dataset directory.

    Returns:
        Path of the written `data.yaml`, which lists `<output_dir>/<split>.txt` per split.
    """
    yaml_lines = [f"path: {json.dumps(str(output_dir))}"]
    for split, image_paths in split_image_paths.items():
        list_path = output_dir / f"{split}.txt"
        list_path.write_text("\n".join(image_paths) + "\n")
        yaml_lines.append(f"{split}: {json.dumps(str(list_path))}")
    yaml_lines += [f"nc: {len(CLASS_NAMES)}", f"names: {json.dumps(list(CLASS_NAMES))}"]
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_path.write_text("\n".join(yaml_lines) + "\n")
    return data_yaml_path


def compute_tile_offsets(length: int, tile_size: int, min_overlap: int) -> list[int]:
    """Place the fewest tiles that cover a frame side with at least `min_overlap` px overlap.

    The tiles are spread evenly, so the first starts at 0 and the last ends at `length`.

    Args:
        length: Frame side in px.
        tile_size: Tile side in px; a side of at most `tile_size` gets one tile.
        min_overlap: Minimum overlap of neighbouring tiles in px, below `tile_size`.

    Returns:
        Start of every tile in px, ascending.
    """
    if length <= tile_size:
        return [0]
    count = math.ceil((length - min_overlap) / (tile_size - min_overlap))
    return np.linspace(0, length - tile_size, count).round().astype(int).tolist()


def clip_boxes_to_tile(
    boxes: np.ndarray, tile: tuple[int, int, int, int], min_visible_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clip frame boxes to a tile and keep those with enough of their area inside it.

    Args:
        boxes: Array of shape (n, 4) with `x1, y1, x2, y2` in px of the frame.
        tile: Tile `x1, y1, x2, y2` in px of the frame.
        min_visible_fraction: Minimum share of a box's area inside the tile, in (0, 1].

    Returns:
        Indices of the kept boxes, and an array of shape (k, 4) with their clipped
        `x, y, w, h` in px of the tile (top-left corner).
    """
    x1, y1, x2, y2 = tile
    clipped = np.column_stack(
        [
            np.clip(boxes[:, 0], x1, x2),
            np.clip(boxes[:, 1], y1, y2),
            np.clip(boxes[:, 2], x1, x2),
            np.clip(boxes[:, 3], y1, y2),
        ]
    )
    sizes = clipped[:, 2:] - clipped[:, :2]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    is_kept = (sizes > 0).all(axis=1) & (sizes.prod(axis=1) >= min_visible_fraction * areas)
    kept = np.flatnonzero(is_kept)
    return kept, np.column_stack([clipped[kept, :2] - [x1, y1], sizes[kept]])


def write_frame_tiles(
    frame: dict,
    objects: pl.DataFrame,
    ignored: pl.DataFrame | None,
    tiles: dict,
    output_dir: Path,
) -> list[str]:
    """Resize one frame to the tile frame width, cut it into tiles, and write their labels.

    With `ignored`, its regions are filled with `tiles.fill` before resizing, except inside
    labelled boxes (`mask_ignored_regions`).

    Args:
        frame: Image row with `clip`, `width`, `height`, and absolute `file_path`.
        objects: Objects of the frame with `class_id` and pixel `x`, `y`, `w`, `h`.
        ignored: Ignored regions of the frame with pixel `x`, `y`, `w`, `h`, or None to keep
            the frame unmodified.
        tiles: `[tiles]` table with `frame_width`, `tile_size`, `min_overlap`,
            `min_visible_fraction`, and `fill`.
        output_dir: Dataset directory; tiles go to `images/<clip>/<frame>_<i>.jpg` and their
            labels to `labels/<clip>/<frame>_<i>.txt`.

    Returns:
        Absolute paths of the written tile images.
    """
    image = decode_image(frame["file_path"], mode=ImageReadMode.RGB)
    boxes = convert_xywh_to_xyxy(objects.select(BOX_COLUMNS).to_numpy().astype(np.float64))
    if ignored is not None:
        ignored_boxes = convert_xywh_to_xyxy(
            ignored.select(BOX_COLUMNS).to_numpy().astype(np.float64)
        )
        image = mask_ignored_regions(
            image, torch.from_numpy(ignored_boxes), torch.from_numpy(boxes), tiles["fill"]
        )
    scale = tiles["frame_width"] / frame["width"]
    height = round(frame["height"] * scale)
    resized = torch.nn.functional.interpolate(
        image[None].float(),
        size=(height, tiles["frame_width"]),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )
    resized = resized[0].round().clamp(0, 255).to(torch.uint8)
    boxes *= scale
    class_ids = objects["class_id"].to_numpy()
    tile_size = tiles["tile_size"]
    stem = Path(frame["file_path"]).stem
    tile_paths = []
    for y in compute_tile_offsets(height, tile_size, tiles["min_overlap"]):
        for x in compute_tile_offsets(tiles["frame_width"], tile_size, tiles["min_overlap"]):
            tile_height, tile_width = min(tile_size, height), min(tile_size, tiles["frame_width"])
            kept, tile_boxes = clip_boxes_to_tile(
                boxes, (x, y, x + tile_width, y + tile_height), tiles["min_visible_fraction"]
            )
            yolo_boxes = convert_xywh_to_yolo(tile_boxes, tile_width, tile_height)
            name = f"{stem}_{len(tile_paths)}"
            image_path = output_dir / "images" / frame["clip"] / f"{name}.jpg"
            label_path = output_dir / "labels" / frame["clip"] / f"{name}.txt"
            write_jpeg(
                resized[:, y : y + tile_height, x : x + tile_width].contiguous(),
                str(image_path),
                quality=TILE_JPEG_QUALITY,
            )
            label_path.write_text(
                "".join(
                    f"{class_id} {' '.join(f'{v:.6f}' for v in box)}\n"
                    for class_id, box in zip(class_ids[kept], yolo_boxes, strict=True)
                )
            )
            tile_paths.append(str(image_path))
    return tile_paths


def write_yolo_tile_dataset(
    annotations: AnnotationTables,
    frame_steps: dict[str, int],
    output_dir: Path,
    tiles: dict,
    masked_splits: set[str],
) -> Path:
    """Write an Ultralytics dataset of overlapping tiles cut from the configured splits.

    Every selected frame is resized to `tiles.frame_width` px wide (keeping its aspect ratio),
    so that objects have the size they get at inference with `imgsz` equal to that width,
    and cut into `tiles.tile_size` px tiles that overlap by at least `tiles.min_overlap` px,
    in the spirit of SAHI slicing (https://arxiv.org/abs/2202.06934). Boxes are clipped to
    each tile and kept if at least `tiles.min_visible_fraction` of their area is inside it.
    Tile images are JPEG files written into `output_dir`, never into the data root.

    Args:
        annotations: Annotation tables of all configured splits.
        frame_steps: Mapping from split name to its frame step, e.g. `{"train": 5, "val": 5}`.
        output_dir: Directory to create, e.g. under `runs/`; it must not exist yet.
        tiles: `[tiles]` table with `frame_width`, `tile_size`, `min_overlap`,
            `min_visible_fraction`, and `fill`.
        masked_splits: Splits whose ignored regions are filled with `tiles.fill`.

    Returns:
        Path of the written `data.yaml`, which lists `<output_dir>/<split>.txt` per split.
    """
    split_frames = {
        split: select_frames(annotations.images.filter(pl.col("split") == split), frame_step)
        .sort("clip", "frame_id")
        .to_dicts()
        for split, frame_step in frame_steps.items()
    }
    for split, frames in split_frames.items():
        if not frames:
            raise ValueError(f"No frames in split {split!r}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True)
    frame_objects = annotations.objects.partition_by(FRAME_KEY, as_dict=True)
    frame_ignored = annotations.ignored.partition_by(FRAME_KEY, as_dict=True)
    empty_objects = annotations.objects.clear()
    split_image_paths = {}
    with ThreadPoolExecutor(max_workers=TILE_WRITER_THREADS) as executor:
        for split, frames in split_frames.items():
            for clip in {frame["clip"] for frame in frames}:
                (output_dir / "images" / clip).mkdir(parents=True, exist_ok=True)
                (output_dir / "labels" / clip).mkdir(parents=True, exist_ok=True)
            keys = [(split, frame["image_id"]) for frame in frames]
            frame_tile_paths = executor.map(
                write_frame_tiles,
                frames,
                [frame_objects.get(key, empty_objects) for key in keys],
                [
                    (
                        frame_ignored.get(key, annotations.ignored.clear())
                        if split in masked_splits
                        else None
                    )
                    for key in keys
                ],
                [tiles] * len(frames),
                [output_dir] * len(frames),
            )
            split_image_paths[split] = [path for paths in frame_tile_paths for path in paths]
    return write_data_yaml(split_image_paths, output_dir)
