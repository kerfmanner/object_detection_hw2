from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from src.data.annotations import AnnotationTables
from src.visualization.frames import (
    draw_class_boxes,
    find_densest_crop,
    plot_frame_with_crop,
    plot_ground_truth_and_predictions,
    render_figure_as_jpeg,
    shade_ignored_regions,
)


def make_boxes(rows: list[tuple[int, int, int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["class_id", "x", "y", "w", "h"], orient="row").with_columns(
        (pl.col("w") * pl.col("h")).alias("area")
    )


def test_find_densest_crop_selects_window_with_most_small_objects() -> None:
    objects = make_boxes(
        [(0, 310, 210, 10, 10), (0, 330, 230, 10, 10), (0, 20, 20, 10, 10), (2, 0, 0, 200, 200)]
    )
    assert find_densest_crop(objects, (400, 300), (100, 100), 32**2) == (250, 150)


def test_find_densest_crop_counts_objects_below_the_given_area() -> None:
    objects = make_boxes([(2, 300, 200, 40, 40), (2, 0, 0, 200, 200)])
    assert find_densest_crop(objects, (400, 300), (100, 100), 32**2) == (0, 0)
    assert find_densest_crop(objects, (400, 300), (100, 100), 50**2) == (250, 150)


def test_drawing_helpers_add_one_patch_per_box() -> None:
    fig, ax = plt.subplots()
    draw_class_boxes(ax, make_boxes([(0, 0, 0, 5, 5), (7, 1, 1, 5, 5)]), 1.0)
    shade_ignored_regions(ax, make_boxes([(0, 0, 0, 5, 5)]))
    assert len(ax.patches) == 3
    plt.close(fig)


def test_plot_frame_with_crop_renders_smoke_frame_as_jpeg(
    smoke_annotations: AnnotationTables,
) -> None:
    image = smoke_annotations.images.row(0, named=True)
    keys = (pl.col("split") == image["split"]) & (pl.col("image_id") == image["image_id"])
    fig = plot_frame_with_crop(
        Path(image["file_path"]),
        smoke_annotations.objects.filter(keys),
        smoke_annotations.ignored.filter(keys),
        (0, 0),
        (image["width"] // 4, image["height"] // 4),
        "smoke frame",
    )
    assert render_figure_as_jpeg(fig, 80)[:2] == b"\xff\xd8"


def test_ground_truth_and_prediction_panels_count_their_boxes(
    smoke_annotations: AnnotationTables,
) -> None:
    image = smoke_annotations.images.row(0, named=True)
    keys = (pl.col("split") == image["split"]) & (pl.col("image_id") == image["image_id"])
    objects = smoke_annotations.objects.filter(keys)
    fig = plot_ground_truth_and_predictions(
        Path(image["file_path"]),
        objects,
        smoke_annotations.ignored.filter(keys),
        make_boxes([(0, 10, 10, 20, 20), (7, 40, 40, 20, 20)]),
        "smoke frame",
    )
    ground_truth_ax, prediction_ax = fig.axes
    assert ground_truth_ax.get_title() == f"Ground truth, {objects.height} boxes"
    assert prediction_ax.get_title() == "Predictions, 2 boxes"
    assert render_figure_as_jpeg(fig, 80)[:2] == b"\xff\xd8"
