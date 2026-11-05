from pathlib import Path

import polars as pl
import pytest

from src.data.annotations import load_test_images
from src.submission.validate import parse_submission, validate_submission
from src.submission.writer import format_submission, write_submission

TEST_IMAGES = pl.DataFrame(
    {
        "image_id": ["clip_a_0000001", "clip_a_0000002", "clip_b_0000001"],
        "width": [1920, 1920, 680],
        "height": [1080, 1080, 382],
    }
)
PREDICTION_SCHEMA = {
    "image_id": pl.String,
    "class_id": pl.Int64,
    "score": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "w": pl.Float64,
    "h": pl.Float64,
}


def build_predictions(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=PREDICTION_SCHEMA, orient="row")


def split_boxes(prediction_string: str) -> list[list[str]]:
    tokens = prediction_string.split()
    return [tokens[i : i + 6] for i in range(0, len(tokens), 6)]


def write_csv(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_one_row_per_test_image_in_order_with_empty_strings() -> None:
    predictions = build_predictions([("clip_b_0000001", 3, 0.5, 10.0, 20.0, 30.0, 40.0)])
    submission = format_submission(predictions, TEST_IMAGES)
    assert submission.columns == ["image_id", "PredictionString"]
    assert submission["image_id"].to_list() == TEST_IMAGES["image_id"].to_list()
    assert submission["image_id"].is_unique().all()
    assert submission["PredictionString"].to_list() == ["", "", "3 0.5 10.0 20.0 30.0 40.0"]


def test_boxes_are_clipped_to_the_frame_and_empty_boxes_dropped() -> None:
    predictions = build_predictions(
        [
            ("clip_b_0000001", 0, 0.9, -10.0, -5.0, 30.0, 25.0),
            ("clip_b_0000001", 1, 0.8, 660.0, 370.0, 50.0, 50.0),
            ("clip_b_0000001", 2, 0.7, 700.0, 10.0, 20.0, 20.0),
            ("clip_b_0000001", 3, 0.6, 100.0, 100.0, 0.001, 10.0),
            ("clip_b_0000001", 4, 0.5, 1.0049, 2.0, 3.333333, 4.0),
        ]
    )
    boxes = split_boxes(format_submission(predictions, TEST_IMAGES)["PredictionString"][2])
    assert boxes == [
        ["0", "0.9", "0.0", "0.0", "20.0", "20.0"],
        ["1", "0.8", "660.0", "370.0", "20.0", "12.0"],
        ["4", "0.5", "1.0", "2.0", "3.34", "4.0"],
    ]


def test_at_most_500_boxes_per_frame_keep_the_highest_scores() -> None:
    rows = [("clip_a_0000001", i % 8, (i + 1) / 600, 1.0, 1.0, 5.0, 5.0) for i in range(600)]
    submission = format_submission(build_predictions(rows), TEST_IMAGES)
    boxes = split_boxes(submission["PredictionString"][0])
    scores = [float(box[1]) for box in boxes]
    assert len(boxes) == 500
    assert scores == sorted(scores, reverse=True)
    assert min(scores) == pytest.approx(101 / 600, abs=1e-5)
    assert {int(box[0]) for box in boxes} == set(range(8))


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        (("clip_a_0000001", 8, 0.5, 1.0, 1.0, 5.0, 5.0), "class_id outside 0-7"),
        (("clip_a_0000001", 0, 1.5, 1.0, 1.0, 5.0, 5.0), "score outside"),
        (("clip_a_0000001", 0, 0.5, 1.0, 1.0, 0.0, 5.0), "non-positive w or h"),
        (("clip_c_0000001", 0, 0.5, 1.0, 1.0, 5.0, 5.0), "unknown image_id"),
        (("clip_a_0000001", 0, 0.5, float("nan"), 1.0, 5.0, 5.0), "non-finite value"),
    ],
)
def test_invalid_predictions_are_rejected(row: tuple, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        format_submission(build_predictions([row]), TEST_IMAGES)


def test_written_submission_validates_and_parses_back(tmp_path: Path) -> None:
    predictions = build_predictions(
        [
            ("clip_a_0000001", 7, 1.0, 1910.0, 1070.0, 30.0, 30.0),
            ("clip_a_0000001", 0, 0.123456, 0.5, 0.25, 10.0, 12.0),
            ("clip_b_0000001", 5, 0.0, 0.0, 0.0, 680.0, 382.0),
        ]
    )
    csv_path = tmp_path / "submission.csv"
    write_submission(predictions, TEST_IMAGES, csv_path)
    assert csv_path.read_text().splitlines()[0] == "image_id,PredictionString"
    assert validate_submission(csv_path, TEST_IMAGES) == 3
    rows, boxes = parse_submission(csv_path)
    assert rows["PredictionString"].to_list()[1] == ""
    assert boxes.filter(pl.col("image_id") == "clip_a_0000001")["class_id"].to_list() == ["7", "0"]


def test_submission_for_the_real_test_list_validates(data_root: Path, tmp_path: Path) -> None:
    test_images = load_test_images(data_root)
    predictions = test_images.head(5).select(
        "image_id",
        pl.lit(2).alias("class_id"),
        pl.lit(0.75).alias("score"),
        pl.lit(10.0).alias("x"),
        pl.lit(10.0).alias("y"),
        pl.lit(40.0).alias("w"),
        pl.lit(30.0).alias("h"),
    )
    csv_path = tmp_path / "submission.csv"
    submission = write_submission(predictions, test_images, csv_path)
    assert submission.height == test_images.height == 3811
    assert (submission["PredictionString"] == "").sum() == 3811 - 5
    assert validate_submission(csv_path, test_images) == 5


VALID_LINES = [
    "image_id,PredictionString",
    "clip_a_0000001,0 0.5 1.0 1.0 5.0 5.0",
    'clip_a_0000002,""',
    "clip_b_0000001,",
]


@pytest.mark.parametrize(
    ("lines", "problem"),
    [
        (["image_id,Prediction", *VALID_LINES[1:]], "has header"),
        ([*VALID_LINES, "clip_a_0000001,"], "duplicate image_id rows"),
        (VALID_LINES[:-1], "missing test image_id"),
        ([*VALID_LINES, "clip_c_0000001,"], "unknown image_id"),
        ([VALID_LINES[0], "clip_a_0000001,0 0.5 1.0 1.0 5.0", *VALID_LINES[2:]], "multiple of 6"),
        ([VALID_LINES[0], "clip_a_0000001,8 0.5 1 1 5 5", *VALID_LINES[2:]], "class outside"),
        ([VALID_LINES[0], "clip_a_0000001,2.5 0.5 1 1 5 5", *VALID_LINES[2:]], "non-integer"),
        ([VALID_LINES[0], "clip_a_0000001,0 1.2 1 1 5 5", *VALID_LINES[2:]], "score outside"),
        ([VALID_LINES[0], "clip_a_0000001,0 0.5 1 1 0 5", *VALID_LINES[2:]], "non-positive"),
        ([VALID_LINES[0], "clip_a_0000001,0 0.5 1 1 x 5", *VALID_LINES[2:]], "unparsable"),
        ([VALID_LINES[0], "clip_a_0000001,0 0.5 1 1 nan 5", *VALID_LINES[2:]], "non-finite"),
        ([VALID_LINES[0], "clip_b_0000001,0 0.5 -1 1 5 5", *VALID_LINES[1:3]], "outside its frame"),
        (
            [VALID_LINES[0], "clip_b_0000001,0 0.5 676 1 5 5", *VALID_LINES[1:3]],
            "outside its frame",
        ),
        (
            [
                VALID_LINES[0],
                "clip_a_0000001," + " ".join(["0 0.5 1 1 5 5"] * 501),
                *VALID_LINES[2:],
            ],
            "more than 500 boxes",
        ),
    ],
)
def test_validator_rejects_invalid_files(tmp_path: Path, lines: list[str], problem: str) -> None:
    csv_path = write_csv(tmp_path / "submission.csv", lines)
    with pytest.raises(ValueError, match=problem):
        validate_submission(csv_path, TEST_IMAGES)


def test_validator_accepts_quoted_and_unquoted_empty_frames(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path / "submission.csv", VALID_LINES)
    assert validate_submission(csv_path, TEST_IMAGES) == 1
