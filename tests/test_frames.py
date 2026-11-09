from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.data.annotations import AnnotationTables
from src.data.classes import CLASS_NAMES
from src.models.rtdetr import QueryInternals
from src.visualization.frames import (
    compute_context_crop,
    draw_boxes,
    draw_class_boxes,
    find_densest_crop,
    plot_crop_panels,
    plot_error_examples,
    plot_frame_sequence,
    plot_frame_with_crop,
    plot_ground_truth_and_predictions,
    plot_query_attention,
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


def test_context_crop_is_centered_on_the_box_and_stays_inside_the_frame() -> None:
    assert compute_context_crop((100, 100, 10, 20), (400, 300), 4.0, 48) == (65, 70, 80)
    assert compute_context_crop((0, 0, 4, 4), (400, 300), 4.0, 48) == (0, 0, 48)
    assert compute_context_crop((380, 280, 20, 20), (400, 300), 4.0, 48) == (320, 220, 80)
    assert compute_context_crop((0, 0, 400, 300), (400, 300), 4.0, 48) == (50, 0, 300)


def test_crop_panels_and_error_examples_draw_every_box(
    smoke_annotations: AnnotationTables,
) -> None:
    image = smoke_annotations.images.row(0, named=True)
    keys = (pl.col("split") == image["split"]) & (pl.col("image_id") == image["image_id"])
    objects = smoke_annotations.objects.filter(keys)
    ignored = smoke_annotations.ignored.filter(keys)
    fig = plot_crop_panels(
        Path(image["file_path"]),
        ignored,
        (0, 0),
        (200, 100),
        [
            ("Ground truth", objects, "#ffff00"),
            ("Proposals", make_boxes([(0, 1, 1, 5, 5)]), "#0000ff"),
        ],
        "smoke crops",
    )
    assert [len(ax.patches) for ax in fig.axes] == [
        objects.height + ignored.height,
        1 + ignored.height,
    ]
    plt.close(fig)
    first = objects.row(0, named=True)
    examples = pl.DataFrame(
        {
            "category": ["background", "background", "miss"],
            "image_id": [image["image_id"]] * 3,
            "class_id": [2, 3, first["class_id"]],
            "score": [0.9, 0.8, None],
            **{column: [10.0, 20.0, float(first[column])] for column in ("x", "y")},
            **{column: [8.0, 8.0, float(first[column])] for column in ("w", "h")},
        }
    )
    detections = examples.filter(pl.col("score").is_not_null())
    fig = plot_error_examples(
        examples, smoke_annotations.images, objects, ignored, detections, 4.0, 64
    )
    drawn = [ax for ax in fig.axes if ax.axison]
    assert [ax.get_title() for ax in drawn] == [
        "background: car 0.90",
        "background: van 0.80",
        f"miss: {CLASS_NAMES[first['class_id']]}",
    ]
    assert all(len(ax.patches) == objects.height + ignored.height + 3 for ax in drawn)
    plt.close(fig)


def test_frame_sequence_draws_the_boxes_of_each_frame_in_its_panel(
    smoke_annotations: AnnotationTables,
) -> None:
    frames = (
        smoke_annotations.images.filter(pl.col("split") == "train")
        .sort("frame_id")
        .head(3)
        .with_columns(panel_title=pl.format("frame {}", pl.col("frame_id")))
    )
    image_ids = frames["image_id"].to_list()
    objects = smoke_annotations.objects.filter(pl.col("image_id").is_in(image_ids))
    detections = make_boxes([(0, 1, 1, 5, 5), (0, 9, 9, 5, 5)]).with_columns(
        image_id=pl.Series(image_ids[:2])
    )
    fig = plot_frame_sequence(frames, objects, detections, (0, 0), (200, 100), "track")
    assert [ax.get_title() for ax in fig.axes] == frames["panel_title"].to_list()
    assert [len(ax.patches) for ax in fig.axes] == [
        objects.filter(pl.col("image_id") == image_id).height + (i < 2)
        for i, image_id in enumerate(image_ids)
    ]
    plt.close(fig)


def test_draw_boxes_adds_one_outline_per_box() -> None:
    fig, ax = plt.subplots()
    draw_boxes(ax, make_boxes([(0, 0, 0, 5, 5), (1, 1, 1, 5, 5)]), "#ffffff", 1.0)
    assert len(ax.patches) == 2 and not any(patch.get_fill() for patch in ax.patches)
    plt.close(fig)


def test_query_attention_draws_one_panel_per_query_and_layer(
    smoke_annotations: AnnotationTables,
) -> None:
    image = smoke_annotations.images.row(0, named=True)
    layers, queries, heads, levels, points = 3, 2, 2, 3, 4
    internals = QueryInternals(
        queries=pl.DataFrame(
            {"query": [0, 1], "class_id": [2, 0], "score": [0.9, 0.4]}
            | {"x": [10.0, 50.0], "y": [10.0, 20.0], "w": [20.0, 8.0], "h": [10.0, 16.0]}
        ),
        embeddings=np.zeros((queries, 4)),
        layer_scores=np.zeros((layers, queries)),
        layer_boxes=np.full((layers, queries, 4), 10.0),
        sampling_points=np.full((layers, queries, heads, levels, points, 2), 15.0),
        attention_weights=np.full((layers, queries, heads, levels, points), 1 / 12),
        level_strides=[8, 16, 32],
    )
    fig = plot_query_attention(Path(image["file_path"]), internals, [1, 0], [0, 2], 4.0, 64)
    assert [ax.get_title() for ax in fig.axes] == [
        "person 0.40, layer 1",
        "person 0.40, layer 3",
        "car 0.90, layer 1",
        "car 0.90, layer 3",
    ]
    assert all(len(ax.patches) == 1 and len(ax.collections) == levels for ax in fig.axes)
    plt.close(fig)
