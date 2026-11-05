import json
import tomllib
from pathlib import Path
from typing import NamedTuple

import polars as pl

from src.data.classes import CLASS_NAMES

BOX_COLUMNS = ("x", "y", "w", "h")


SOURCE_SPLITS = ("train", "val")


class AnnotationTables(NamedTuple):
    """Annotations as Polars tables keyed by (`split`, `image_id`).

    `split` is the assigned split from the split configuration and `image_id` is unique
    within it; `source_split` and `source_image_id` locate the frame in the provided
    `<source_split>/annotations_coco.json`.

    Attributes:
        images: One row per frame with `split`, `image_id`, `source_split`, `source_image_id`,
            `clip`, `frame_id`, `width`, `height`, and absolute `file_path`.
        objects: One row per labelled object with `split`, `image_id`, 0-based `class_id`,
            pixel box `x`, `y`, `w`, `h` (top-left corner), `area`, `truncation`, `occlusion`,
            `track_id`, and `orig_category_id`.
        ignored: One row per unique ignored region (`iscrowd=1`) with `split`, `image_id`,
            pixel box `x`, `y`, `w`, `h`, `area`, and `orig_category_id`.
    """

    images: pl.DataFrame
    objects: pl.DataFrame
    ignored: pl.DataFrame


def validate_categories(categories: list[dict], annotations_path: Path) -> None:
    """Check that the COCO categories are 1-based ids in the order of `CLASS_NAMES`.

    Args:
        categories: The `categories` list of a COCO annotation file.
        annotations_path: Path of the file, used in the error message.
    """
    expected = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, start=1)]
    if categories != expected:
        raise ValueError(f"Unexpected categories in {annotations_path}: {categories}")


def load_source_annotations(data_root: Path, source_split: str) -> AnnotationTables:
    """Read one provided `<source_split>/annotations_coco.json` into tables.

    The source file repeats every ignored region once per category so that COCO evaluation
    ignores it for each class; the returned `ignored` table keeps one row per region.

    Args:
        data_root: Dataset root containing one directory per provided split.
        source_split: Provided split directory name, `train` or `val`.

    Returns:
        Tables keyed by (`source_split`, `source_image_id`) instead of (`split`, `image_id`);
        `images` has no `split` and `image_id` columns yet.
    """
    split_dir = data_root / source_split
    annotations_path = split_dir / "annotations_coco.json"
    coco = json.loads(annotations_path.read_text())
    validate_categories(coco["categories"], annotations_path)
    source_key = [pl.lit(source_split).alias("source_split")]

    images = pl.DataFrame(coco["images"]).select(
        *source_key,
        pl.col("id").alias("source_image_id"),
        pl.col("video").alias("clip"),
        "frame_id",
        "width",
        "height",
        pl.concat_str(pl.lit(f"{split_dir}/"), pl.col("file_name")).alias("file_path"),
    )
    annotations = (
        pl.DataFrame(coco["annotations"], infer_schema_length=None)
        .rename({"image_id": "source_image_id"})
        .with_columns(pl.col("bbox").list.get(i).alias(name) for i, name in enumerate(BOX_COLUMNS))
    )
    invalid_crowd = annotations.filter(~pl.col("iscrowd").is_in([0, 1])).height
    if invalid_crowd:
        raise ValueError(f"{invalid_crowd} annotations with iscrowd not in {{0, 1}}")

    objects = annotations.filter(pl.col("iscrowd") == 0).select(
        *source_key,
        "source_image_id",
        (pl.col("category_id") - 1).alias("class_id"),
        *BOX_COLUMNS,
        "area",
        "truncation",
        "occlusion",
        "track_id",
        "orig_category_id",
    )
    ignored = (
        annotations.filter(pl.col("iscrowd") == 1)
        .select(*source_key, "source_image_id", *BOX_COLUMNS, "area", "orig_category_id")
        .unique(maintain_order=True)
    )
    return AnnotationTables(images=images, objects=objects, ignored=ignored)


def assign_splits(images: pl.DataFrame, split_clips: dict[str, list[str]]) -> pl.DataFrame:
    """Assign every frame to its configured split and number frames within each split.

    Args:
        images: Image table with `clip` and `frame_id` from the provided source files.
        split_clips: Mapping from split name to the clips it contains.

    Returns:
        The image table with `split` and `image_id` (1-based, ordered by clip and frame)
        as the first columns.
    """
    clip_splits = pl.DataFrame(
        [(clip, split) for split, clips in split_clips.items() for clip in clips],
        schema=["clip", "split"],
        orient="row",
    )
    duplicated = clip_splits.filter(pl.col("clip").is_duplicated())["clip"].unique().to_list()
    unassigned = sorted(set(images["clip"]) - set(clip_splits["clip"]))
    unknown = sorted(set(clip_splits["clip"]) - set(images["clip"]))
    if duplicated or unassigned or unknown:
        raise ValueError(
            f"Invalid split configuration: duplicated {duplicated}, unassigned {unassigned}, "
            f"unknown {unknown}"
        )
    return (
        images.join(clip_splits, on="clip")
        .sort("split", "clip", "frame_id")
        .with_columns(pl.int_range(1, pl.len() + 1).over("split").alias("image_id"))
        .select("split", "image_id", pl.exclude("split", "image_id"))
    )


def load_annotations(data_root: Path, split_clips: dict[str, list[str]]) -> AnnotationTables:
    """Read both provided annotation files and assign frames to the configured splits.

    Args:
        data_root: Dataset root containing one directory per provided split.
        split_clips: Mapping from split name to clip names, e.g. from `load_split_clips`.

    Returns:
        Tables of all configured splits; join on (`split`, `image_id`).
    """
    sources = [load_source_annotations(data_root, source) for source in SOURCE_SPLITS]
    images = assign_splits(pl.concat([tables.images for tables in sources]), split_clips)
    source_key = ["source_split", "source_image_id"]
    frame_keys = images.select("split", "image_id", *source_key)
    objects, ignored = (
        pl.concat(tables)
        .with_row_index("source_order")
        .join(frame_keys, on=source_key)
        .sort("split", "image_id", "source_order")
        .select("split", "image_id", pl.exclude("split", "image_id", "source_order", *source_key))
        for tables in ([t.objects for t in sources], [t.ignored for t in sources])
    )
    return AnnotationTables(images=images, objects=objects, ignored=ignored)


def load_split_clips(data_config_path: Path) -> dict[str, list[str]]:
    """Read the clip lists of the `[splits]` table of the data configuration.

    Args:
        data_config_path: Path of `config/data.toml`.

    Returns:
        Mapping from split name (`train`, `val`) to its clip names.
    """
    splits = tomllib.loads(data_config_path.read_text())["splits"]
    return {split: list(splits[split]) for split in SOURCE_SPLITS}


TEST_IMAGES_COLUMNS = ("image_id", "file_name", "clip", "frame", "width", "height")


def load_test_images(data_root: Path) -> pl.DataFrame:
    """Read `test_images.csv` into an image table shaped like `AnnotationTables.images`.

    Args:
        data_root: Dataset root containing `test_images.csv` and `test/`.

    Returns:
        One row per test frame with `split` (`test`), string `image_id` used in submissions,
        `clip`, `frame_id`, `width`, `height`, and absolute `file_path`.
    """
    csv_path = data_root / "test_images.csv"
    test_images = pl.read_csv(csv_path)
    if tuple(test_images.columns) != TEST_IMAGES_COLUMNS:
        raise ValueError(f"Unexpected columns in {csv_path}: {test_images.columns}")
    return test_images.select(
        pl.lit("test").alias("split"),
        "image_id",
        "clip",
        pl.col("frame").alias("frame_id"),
        "width",
        "height",
        pl.concat_str(pl.lit(f"{data_root}/"), pl.col("file_name")).alias("file_path"),
    )
