import json
import shutil
from pathlib import Path

import polars as pl

from src.data.classes import CLASS_NAMES
from src.data.dataset import select_frames


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
    yaml_lines = [f"path: {json.dumps(str(output_dir))}"]
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
        list_path = output_dir / f"{split}.txt"
        list_path.write_text("\n".join(export_paths) + "\n")
        yaml_lines.append(f"{split}: {json.dumps(str(list_path))}")
    yaml_lines += [f"nc: {len(CLASS_NAMES)}", f"names: {json.dumps(list(CLASS_NAMES))}"]
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_path.write_text("\n".join(yaml_lines) + "\n")
    return data_yaml_path
