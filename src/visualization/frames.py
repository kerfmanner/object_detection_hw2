import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from src.data.annotations import BOX_COLUMNS
from src.data.classes import CLASS_NAMES
from src.models.rtdetr import QueryInternals
from src.visualization.style import (
    CLASS_COLORS,
    FEATURE_LEVEL_COLORS,
    GT_COLOR,
    IGNORED_REGION_ALPHA,
    IGNORED_REGION_COLOR,
    PREDICTION_COLOR,
)

EXAMPLE_LINEWIDTH = 2.5
CONTEXT_LINEWIDTH = 0.9


def draw_class_boxes(ax: Axes, objects: pl.DataFrame, linewidth: float) -> None:
    """Draw box outlines colored by class.

    Args:
        ax: Axes showing the frame in pixel coordinates.
        objects: Object table with `class_id` and pixel box columns.
        linewidth: Outline width in points.
    """
    for class_id, x, y, w, h in objects.select("class_id", *BOX_COLUMNS).iter_rows():
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor=CLASS_COLORS[CLASS_NAMES[class_id]],
                linewidth=linewidth,
            )
        )


def draw_boxes(ax: Axes, boxes: pl.DataFrame, color: str, linewidth: float) -> None:
    """Draw box outlines in one color.

    Args:
        ax: Axes showing the frame in pixel coordinates.
        boxes: Table with pixel box columns.
        color: Outline color.
        linewidth: Outline width in points.
    """
    for x, y, w, h in boxes.select(BOX_COLUMNS).iter_rows():
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=linewidth))


def shade_ignored_regions(ax: Axes, ignored: pl.DataFrame) -> None:
    """Shade ignored regions with the shared ignored-region color.

    Args:
        ax: Axes showing the frame in pixel coordinates.
        ignored: Ignored-region table with pixel box columns.
    """
    for x, y, w, h in ignored.select(BOX_COLUMNS).iter_rows():
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=IGNORED_REGION_COLOR,
                alpha=IGNORED_REGION_ALPHA,
                edgecolor="none",
            )
        )


def find_densest_crop(
    objects: pl.DataFrame,
    frame_size: tuple[int, int],
    crop_size: tuple[int, int],
    max_area: float,
) -> tuple[int, int]:
    """Find the crop window that contains the most centers of objects below an area limit.

    Windows are searched on a grid with half the crop size as stride; ties keep the first
    window in row-major order.

    Args:
        objects: Object table with pixel box columns and `area`.
        frame_size: Frame (width, height) in px.
        crop_size: Crop (width, height) in px, at most the frame size.
        max_area: Objects with `area` below this value in px² are counted.

    Returns:
        Top-left corner (x, y) of the selected crop in px.
    """
    small = objects.filter(pl.col("area") < max_area)
    centers_x = (small["x"] + small["w"] / 2).to_numpy()
    centers_y = (small["y"] + small["h"] / 2).to_numpy()
    starts = [
        np.unique(np.append(np.arange(0, frame - crop + 1, max(crop // 2, 1)), frame - crop))
        for frame, crop in zip(frame_size, crop_size, strict=True)
    ]
    best_count, best_corner = -1, (0, 0)
    for y in starts[1]:
        for x in starts[0]:
            count = int(
                np.sum(
                    (centers_x >= x)
                    & (centers_x < x + crop_size[0])
                    & (centers_y >= y)
                    & (centers_y < y + crop_size[1])
                )
            )
            if count > best_count:
                best_count, best_corner = count, (int(x), int(y))
    return best_corner


def add_class_legend(fig: Figure) -> None:
    """Add a one-row legend of the class colors and the ignored-region shading below a figure.

    Args:
        fig: Figure whose bottom margin holds the legend.
    """
    handles = [
        Patch(edgecolor=color, fill=False, label=name) for name, color in CLASS_COLORS.items()
    ]
    handles.append(
        Patch(color=IGNORED_REGION_COLOR, alpha=IGNORED_REGION_ALPHA, label="ignored region")
    )
    fig.legend(handles=handles, ncol=9, loc="lower center", fontsize=8, frameon=False)


def plot_frame_with_crop(
    image_path: Path,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    crop_corner: tuple[int, int],
    crop_size: tuple[int, int],
    title: str,
) -> Figure:
    """Plot a frame with GT boxes and ignored regions next to a zoomed crop of it.

    Args:
        image_path: Path of the frame image.
        objects: Objects of this frame with `class_id` and pixel box columns.
        ignored: Ignored regions of this frame with pixel box columns.
        crop_corner: Top-left corner (x, y) of the crop in px.
        crop_size: Crop (width, height) in px.
        title: Figure title.

    Returns:
        Figure with the full frame (crop outlined) on the left and the crop on the right.
    """
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    x0, y0 = crop_corner
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    for ax, linewidth in zip(axes, (0.6, 1.4)):
        ax.imshow(image)
        shade_ignored_regions(ax, ignored)
        draw_class_boxes(ax, objects, linewidth)
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
    axes[0].set(xlim=(0, width), ylim=(height, 0), title=f"Full frame, {width} × {height} px")
    axes[0].add_patch(Rectangle((x0, y0), *crop_size, fill=False, edgecolor="white", linewidth=1.5))
    axes[1].set(
        xlim=(x0, x0 + crop_size[0]),
        ylim=(y0 + crop_size[1], y0),
        title=f"Crop {crop_size[0]} × {crop_size[1]} px at ({x0}, {y0})",
    )
    add_class_legend(fig)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_ground_truth_and_predictions(
    image_path: Path,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    predictions: pl.DataFrame,
    title: str,
) -> Figure:
    """Plot a frame with its GT boxes next to the same frame with predicted boxes.

    Both panels shade the ignored regions and color boxes by class.

    Args:
        image_path: Path of the frame image.
        objects: GT objects of this frame with `class_id` and pixel box columns.
        ignored: Ignored regions of this frame with pixel box columns.
        predictions: Predictions to draw on this frame with `class_id` and pixel box columns.
        title: Figure title.

    Returns:
        Figure with the GT panel on the left and the prediction panel on the right.
    """
    image = plt.imread(image_path)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    panels = (
        (objects, f"Ground truth, {objects.height} boxes"),
        (predictions, f"Predictions, {predictions.height} boxes"),
    )
    for ax, (boxes, panel_title) in zip(axes, panels, strict=True):
        ax.imshow(image)
        shade_ignored_regions(ax, ignored)
        draw_class_boxes(ax, boxes, linewidth=0.8)
        ax.set(xticks=[], yticks=[], title=panel_title)
        ax.grid(False)
    add_class_legend(fig)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def render_figure_as_jpeg(fig: Figure, quality: int) -> bytes:
    """Render a figure to JPEG bytes and close it.

    Args:
        fig: Figure to render.
        quality: JPEG quality from 1 to 95.

    Returns:
        Encoded JPEG image.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="jpeg", pil_kwargs={"quality": quality})
    plt.close(fig)
    return buffer.getvalue()


def plot_crop_panels(
    image_path: Path,
    ignored: pl.DataFrame,
    crop_corner: tuple[int, int],
    crop_size: tuple[int, int],
    panels: list[tuple[str, pl.DataFrame, str]],
    title: str,
) -> Figure:
    """Plot the same crop of a frame once per box set, side by side.

    Args:
        image_path: Path of the frame image.
        ignored: Ignored regions of this frame with pixel box columns, shaded in every panel.
        crop_corner: Top-left corner (x, y) of the crop in px.
        crop_size: Crop (width, height) in px.
        panels: Per panel, its title, a table with pixel box columns, and the box color.
        title: Figure title.

    Returns:
        Figure with one axes per panel.
    """
    image = plt.imread(image_path)
    x0, y0 = crop_corner
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False)
    for ax, (panel_title, boxes, color) in zip(axes[0], panels, strict=True):
        ax.imshow(image)
        shade_ignored_regions(ax, ignored)
        draw_boxes(ax, boxes, color, linewidth=CONTEXT_LINEWIDTH)
        ax.set(
            xlim=(x0, x0 + crop_size[0]),
            ylim=(y0 + crop_size[1], y0),
            xticks=[],
            yticks=[],
            title=panel_title,
        )
        ax.grid(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_frame_sequence(
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    detections: pl.DataFrame,
    crop_corner: tuple[int, int],
    crop_size: tuple[int, int],
    title: str,
) -> Figure:
    """Plot the same crop of consecutive frames side by side, with GT and detection boxes.

    Args:
        frames: One row per panel, in panel order, with `image_id`, absolute `file_path`, and
            `panel_title`.
        objects: GT boxes with `image_id` and pixel box columns, drawn thin in the GT color.
        detections: Boxes with `image_id` and pixel box columns, drawn thick in the
            prediction color.
        crop_corner: Top-left corner (x, y) of the crop in px.
        crop_size: Crop (width, height) in px.
        title: Figure title.

    Returns:
        Figure with one axes per frame.
    """
    x0, y0 = crop_corner
    fig, axes = plt.subplots(1, frames.height, figsize=(3 * frames.height, 3.3), squeeze=False)
    for ax, (image_id, file_path, panel_title) in zip(
        axes[0], frames.select("image_id", "file_path", "panel_title").iter_rows(), strict=True
    ):
        ax.imshow(plt.imread(file_path))
        draw_boxes(ax, objects.filter(pl.col("image_id") == image_id), GT_COLOR, CONTEXT_LINEWIDTH)
        draw_boxes(
            ax,
            detections.filter(pl.col("image_id") == image_id),
            PREDICTION_COLOR,
            EXAMPLE_LINEWIDTH,
        )
        ax.set(
            xlim=(x0, x0 + crop_size[0]),
            ylim=(y0 + crop_size[1], y0),
            xticks=[],
            yticks=[],
            title=panel_title,
        )
        ax.grid(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def compute_context_crop(
    box: tuple[float, float, float, float],
    frame_size: tuple[int, int],
    context_scale: float,
    min_side: int,
) -> tuple[int, int, int]:
    """Compute a square crop centered on a box, shifted inside the frame.

    Args:
        box: Box `x, y, w, h` in px.
        frame_size: Frame (width, height) in px.
        context_scale: Crop side as a multiple of the longer box side.
        min_side: Smallest crop side in px.

    Returns:
        Top-left corner x, y and side of the crop in px; the side is at most the shorter
        frame side.
    """
    x, y, w, h = box
    width, height = frame_size
    side = int(min(max(context_scale * max(w, h), min_side), width, height))
    x0 = int(min(max(x + w / 2 - side / 2, 0), width - side))
    y0 = int(min(max(y + h / 2 - side / 2, 0), height - side))
    return x0, y0, side


def plot_error_examples(
    examples: pl.DataFrame,
    images: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    detections: pl.DataFrame,
    context_scale: float,
    min_side: int,
) -> Figure:
    """Plot one zoomed crop per example detection or missed object, one row per category.

    Each crop shows the GT boxes, the kept detections, and the shaded ignored regions of
    its frame; the example itself is drawn with a thick outline, in the prediction color
    for detections and in the GT color for missed objects.

    Args:
        examples: Result of `select_error_examples`.
        images: Image table with `image_id`, `width`, `height`, and absolute `file_path`.
        objects: GT objects with `image_id`, `class_id`, and pixel box columns.
        ignored: Ignored regions with `image_id` and pixel box columns.
        detections: Kept detections with `image_id` and pixel box columns.
        context_scale: Crop side as a multiple of the longer side of the example box.
        min_side: Smallest crop side in px.

    Returns:
        Figure with one row per category and one column per example of the fullest
        category; unused axes are hidden.
    """
    categories = examples["category"].unique(maintain_order=True).to_list()
    columns = max(examples.group_by("category").len()["len"].to_list())
    fig, axes = plt.subplots(
        len(categories), columns, figsize=(2.6 * columns, 2.9 * len(categories)), squeeze=False
    )
    for ax in axes.flat:
        ax.set_axis_off()
    frame_rows = {row["image_id"]: row for row in images.iter_rows(named=True)}
    for row_axes, category in zip(axes, categories, strict=True):
        rows = examples.filter(pl.col("category") == category).iter_rows(named=True)
        for ax, example in zip(row_axes, rows):
            frame = frame_rows[example["image_id"]]
            box = tuple(example[column] for column in BOX_COLUMNS)
            x0, y0, side = compute_context_crop(
                box, (frame["width"], frame["height"]), context_scale, min_side
            )
            is_frame = pl.col("image_id") == example["image_id"]
            ax.set_axis_on()
            ax.imshow(plt.imread(frame["file_path"]))
            shade_ignored_regions(ax, ignored.filter(is_frame))
            draw_boxes(ax, objects.filter(is_frame), GT_COLOR, CONTEXT_LINEWIDTH)
            draw_boxes(ax, detections.filter(is_frame), PREDICTION_COLOR, CONTEXT_LINEWIDTH)
            example_color = GT_COLOR if example["score"] is None else PREDICTION_COLOR
            x, y, w, h = box
            ax.add_patch(
                Rectangle(
                    (x, y), w, h, fill=False, edgecolor=example_color, linewidth=EXAMPLE_LINEWIDTH
                )
            )
            score = "" if example["score"] is None else f" {example['score']:.2f}"
            ax.set(
                xlim=(x0, x0 + side),
                ylim=(y0 + side, y0),
                xticks=[],
                yticks=[],
                title=f"{category}: {CLASS_NAMES[example['class_id']]}{score}",
            )
            ax.title.set_fontsize(9)
            ax.grid(False)
    handles = [
        Patch(edgecolor=GT_COLOR, fill=False, label="ground truth"),
        Patch(edgecolor=PREDICTION_COLOR, fill=False, label="prediction"),
        Patch(color=IGNORED_REGION_COLOR, alpha=IGNORED_REGION_ALPHA, label="ignored region"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_query_attention(
    image_path: Path,
    internals: QueryInternals,
    query_ids: list[int],
    layers: list[int],
    context_scale: float,
    min_side: int,
) -> Figure:
    """Plot the cross-attention sampling locations of queries, one row per query.

    Each panel crops the frame around the query's final box and shows the box of that
    decoder layer and every sampling location of all heads, colored by feature level, with
    an area proportional to its attention weight.

    Args:
        image_path: Path of the frame image.
        internals: Result of `predict_query_internals` for this frame.
        query_ids: Queries to plot, one row each.
        layers: Decoder layers (from 0) to plot, one column each.
        context_scale: Crop side as a multiple of the longer side of the final box.
        min_side: Smallest crop side in px.

    Returns:
        Figure with one axes per query and layer and a legend of the feature levels.
    """
    image = plt.imread(image_path)
    frame_size = (image.shape[1], image.shape[0])
    fig, axes = plt.subplots(
        len(query_ids),
        len(layers),
        figsize=(2.8 * len(layers), 2.9 * len(query_ids)),
        squeeze=False,
    )
    for row_axes, query in zip(axes, query_ids, strict=True):
        final = internals.queries.row(query, named=True)
        x0, y0, side = compute_context_crop(
            (final["x"], final["y"], final["w"], final["h"]), frame_size, context_scale, min_side
        )
        for ax, layer in zip(row_axes, layers, strict=True):
            ax.imshow(image)
            x, y, w, h = internals.layer_boxes[layer, query]
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=False,
                    edgecolor=PREDICTION_COLOR,
                    linewidth=EXAMPLE_LINEWIDTH,
                )
            )
            for level, color in enumerate(FEATURE_LEVEL_COLORS[: len(internals.level_strides)]):
                points = internals.sampling_points[layer, query, :, level].reshape(-1, 2)
                weights = internals.attention_weights[layer, query, :, level].ravel()
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=400 * weights,
                    color=color,
                    alpha=0.7,
                    linewidths=0,
                )
            ax.set(
                xlim=(x0, x0 + side),
                ylim=(y0 + side, y0),
                xticks=[],
                yticks=[],
                title=f"{CLASS_NAMES[final['class_id']]} {final['score']:.2f}, layer {layer + 1}",
            )
            ax.title.set_fontsize(9)
            ax.grid(False)
    fig.legend(
        handles=[
            Patch(color=color, label=f"stride {stride}")
            for stride, color in zip(internals.level_strides, FEATURE_LEVEL_COLORS, strict=False)
        ],
        loc="lower center",
        ncol=len(internals.level_strides),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
