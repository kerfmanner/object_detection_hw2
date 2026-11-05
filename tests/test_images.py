from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from src.data.images import (
    compute_color_statistics,
    compute_frame_differences,
    read_thumbnail,
    sample_frame_pairs,
)


@pytest.fixture
def frame_paths(tmp_path: Path) -> list[str]:
    paths = []
    for name, value in (("black", 0.0), ("gray", 0.5), ("white", 1.0)):
        path = tmp_path / f"{name}.png"
        plt.imsave(path, np.full((16, 24, 3), value))
        paths.append(str(path))
    return paths


def test_read_thumbnail_subsamples_and_scales(frame_paths: list[str]) -> None:
    thumbnail = read_thumbnail(Path(frame_paths[2]), 4)
    assert thumbnail.shape == (4, 6, 3)
    assert thumbnail.max() <= 1.0


def test_compute_color_statistics_measures_uniform_frames(frame_paths: list[str]) -> None:
    statistics = compute_color_statistics(frame_paths, 2)
    assert statistics["luminance_mean"].to_list() == pytest.approx([0.0, 0.5, 1.0], abs=0.01)
    assert statistics["luminance_std"].to_list() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_compute_frame_differences_is_zero_for_identical_frames(frame_paths: list[str]) -> None:
    pairs = pl.DataFrame(
        {"file_path": [frame_paths[0], frame_paths[0]], "file_path_later": frame_paths[:2]}
    )
    differences = compute_frame_differences(pairs, 2)["mean_abs_difference"].to_list()
    assert differences == pytest.approx([0.0, 0.5], abs=0.01)


def test_sample_frame_pairs_spreads_starts_and_skips_short_clips() -> None:
    frames = pl.DataFrame(
        {
            "split": ["train"] * 21 + ["test"] * 3,
            "clip": ["a"] * 21 + ["b"] * 3,
            "frame_id": list(range(1, 22)) + [1, 2, 3],
        }
    ).with_columns(pl.format("{}/{}.jpg", "clip", "frame_id").alias("file_path"))
    pairs = sample_frame_pairs(frames, (1, 5), 3)
    assert pairs["clip"].unique().to_list() == ["a"]
    assert sorted(pairs.filter(pl.col("offset") == 5)["frame_id"].to_list()) == [1, 8, 16]
    row = pairs.filter(pl.col("offset") == 5, pl.col("frame_id") == 16).row(0, named=True)
    assert (row["file_path"], row["file_path_later"]) == ("a/16.jpg", "a/21.jpg")
