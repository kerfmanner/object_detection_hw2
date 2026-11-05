import polars as pl
import pytest

from src.data.tracks import pair_track_boxes, summarize_tracks


@pytest.fixture
def tracked_objects() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "split": ["train"] * 5,
            "clip": ["a", "a", "a", "b", "a"],
            "track_id": [1, 1, 1, 1, 2],
            "frame_id": [1, 2, 4, 2, 1],
            "class_id": [2, 2, 3, 0, 0],
            "x": [0, 3, 10, 0, 50],
            "y": [0, 4, 0, 0, 50],
            "w": [10, 10, 10, 10, 4],
            "h": [10, 10, 10, 10, 4],
        }
    )


def test_pair_track_boxes_pairs_same_track_and_clip_only(tracked_objects: pl.DataFrame) -> None:
    pairs = pair_track_boxes(tracked_objects, 1)
    assert pairs.height == 1
    row = pairs.row(0, named=True)
    assert (row["clip"], row["track_id"], row["offset"]) == ("a", 1, 1)
    assert row["displacement"] == pytest.approx(5.0)
    assert row["relative_displacement"] == pytest.approx(0.5)
    assert row["iou"] == pytest.approx(42 / 158)


def test_pair_track_boxes_skips_missing_frames(tracked_objects: pl.DataFrame) -> None:
    pairs = pair_track_boxes(tracked_objects, 2)
    assert pairs.height == 1
    assert pairs["iou"].item() == pytest.approx(18 / 182)
    assert pair_track_boxes(tracked_objects, 5).is_empty()


def test_summarize_tracks_reports_length_span_and_gaps(tracked_objects: pl.DataFrame) -> None:
    tracks = summarize_tracks(tracked_objects)
    first = tracks.filter(pl.col("clip") == "a", pl.col("track_id") == 1).row(0, named=True)
    assert (first["class_id"], first["class_count"], first["frames"], first["span"]) == (2, 2, 3, 4)
    assert first["has_gap"]
    assert tracks.height == 3
