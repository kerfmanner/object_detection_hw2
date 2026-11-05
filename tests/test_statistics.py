import polars as pl
import pytest

from src.data.classes import CLASS_NAMES
from src.data.statistics import (
    assign_size_bucket,
    compute_clip_class_shares,
    compute_required_max_size,
    compute_resize_scale,
    compute_resized_frame_size,
    count_class_presence,
    count_objects_per_frame,
    summarize_quantiles,
)


@pytest.fixture
def toy_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    images = pl.DataFrame(
        {
            "split": ["train", "train", "train", "val"],
            "image_id": [1, 2, 3, 1],
            "clip": ["a", "a", "b", "c"],
        }
    )
    objects = pl.DataFrame(
        {
            "split": ["train", "train", "train", "train", "val"],
            "image_id": [1, 1, 2, 3, 1],
            "class_id": [0, 0, 2, 0, 2],
        }
    )
    return images, objects


def test_count_class_presence_counts_instances_frames_and_clips(
    toy_tables: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    presence = count_class_presence(*toy_tables)
    assert presence.height == 2 * len(CLASS_NAMES)
    person = presence.filter(pl.col("split") == "train", pl.col("class_id") == 0).row(0, named=True)
    assert (person["instances"], person["frames"], person["clips"]) == (3, 2, 2)
    assert person["instance_share"] == pytest.approx(0.75)
    assert person["frame_share"] == pytest.approx(2 / 3)
    assert person["clip_share"] == pytest.approx(1.0)
    absent = presence.filter(pl.col("split") == "val", pl.col("class_id") == 0).row(0, named=True)
    assert (absent["instances"], absent["frames"], absent["clips"]) == (0, 0, 0)


def test_compute_clip_class_shares_sums_to_one_per_clip(
    toy_tables: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    shares = compute_clip_class_shares(*toy_tables)
    assert shares["clip"].to_list() == ["a", "b", "c"]
    assert shares.columns == ["split", "clip", "objects", *CLASS_NAMES]
    assert shares["objects"].to_list() == [3, 1, 1]
    assert shares.select(pl.sum_horizontal(CLASS_NAMES)).to_series().to_list() == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert shares.filter(pl.col("clip") == "a")["person"].item() == pytest.approx(2 / 3)


def test_assign_size_bucket_uses_coco_limits() -> None:
    areas = pl.DataFrame({"area": [1, 32**2 - 1, 32**2, 96**2 - 1, 96**2, 10**6]})
    buckets = areas.select(assign_size_bucket(pl.col("area"))).to_series().cast(pl.String)
    assert buckets.to_list() == ["small", "small", "medium", "medium", "large", "large"]


def test_resize_scale_keeps_shorter_side_unless_max_size_binds() -> None:
    frames = pl.DataFrame({"width": [1920, 1080, 4000], "height": [1080, 1920, 1000]})
    scales = frames.select(compute_resize_scale(1024, 2000)).to_series().to_list()
    assert scales == pytest.approx([1024 / 1080, 1024 / 1080, 2000 / 4000])


def test_required_max_size_keeps_shorter_side_at_min_size() -> None:
    frames = pl.DataFrame({"width": [1920, 680, 1000], "height": [1080, 382, 1000]})
    required = frames.select(compute_required_max_size(1024)).to_series()
    assert required.to_list() == [1821, 1823, 1024]
    scales = frames.with_columns(required.alias("max_size")).select(
        pl.min_horizontal(
            1024 / pl.min_horizontal("width", "height"),
            pl.col("max_size") / pl.max_horizontal("width", "height"),
        )
        * pl.min_horizontal("width", "height")
    )
    assert scales.to_series().to_list() == pytest.approx([1024, 1024, 1024])


def test_summarize_quantiles_reports_count_mean_and_quantiles() -> None:
    table = pl.DataFrame({"group": ["a"] * 5 + ["b"], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]})
    summary = summarize_quantiles(table, "value", ["group"], (0.5, 0.95))
    assert summary.columns == ["group", "count", "mean", "p50", "p95"]
    row = summary.row(0, named=True)
    assert (row["count"], row["mean"], row["p50"]) == (5, 3.0, 3.0)
    assert row["p95"] == pytest.approx(4.8)


@pytest.mark.parametrize(
    ("max_size", "expected"),
    [
        (1821, [(1820, 1024), (1820, 1023), (1820, 1023), (1820, 1022), (1818, 1024)]),
        (1823, [(1820, 1024), (1820, 1023), (1820, 1023), (1822, 1024), (1818, 1024)]),
    ],
)
def test_resized_frame_size_matches_torchvision_outputs(
    max_size: int, expected: list[tuple[int, int]]
) -> None:
    frames = pl.DataFrame(
        {"width": [1920, 2720, 1360, 680, 1904], "height": [1080, 1530, 765, 382, 1072]}
    )
    resized = frames.select(compute_resized_frame_size(1024, max_size)).rows()
    assert resized == expected


def test_count_objects_per_frame_keeps_empty_frames(
    toy_tables: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    images, objects = toy_tables
    empty_frame = pl.DataFrame({"split": ["val"], "image_id": [2], "clip": ["c"]})
    counts = count_objects_per_frame(pl.concat([images, empty_frame]), objects)
    assert counts.sort("split", "image_id")["objects"].to_list() == [2, 1, 1, 1, 0]
