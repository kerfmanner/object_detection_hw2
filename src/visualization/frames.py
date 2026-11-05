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
from src.visualization.style import CLASS_COLORS, IGNORED_REGION_ALPHA, IGNORED_REGION_COLOR


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
    handles = [
        Patch(edgecolor=color, fill=False, label=name) for name, color in CLASS_COLORS.items()
    ]
    handles.append(
        Patch(color=IGNORED_REGION_COLOR, alpha=IGNORED_REGION_ALPHA, label="ignored region")
    )
    fig.legend(handles=handles, ncol=9, loc="lower center", fontsize=8, frameon=False)
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
