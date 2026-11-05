from pathlib import Path

import polars as pl

from src.data.annotations import load_test_images


def test_test_images_have_unique_ids_and_existing_files(data_root: Path) -> None:
    test_images = load_test_images(data_root)
    assert test_images.height == 3811
    assert test_images["image_id"].n_unique() == test_images.height
    assert test_images["clip"].n_unique() == 48
    assert test_images.filter((pl.col("width") <= 0) | (pl.col("height") <= 0)).is_empty()
    first_frames = test_images.group_by("clip").agg(pl.col("file_path").first())
    assert all(Path(path).is_file() for path in first_frames["file_path"])
