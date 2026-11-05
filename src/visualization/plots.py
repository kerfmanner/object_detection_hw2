import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from src.data.classes import CLASS_NAMES
from src.visualization.style import SPLIT_COLORS


def plot_class_histograms(
    objects: pl.DataFrame, column: str, bins: np.ndarray, xlabel: str, title: str
) -> Figure:
    """Plot one histogram panel per class with train and val overlaid as densities.

    Args:
        objects: Object table with `split`, `class_id`, and `column`.
        column: Numeric column to plot.
        bins: Bin edges; a logarithmic x axis is used.
        xlabel: Label of the shared x axis, with units.
        title: Figure title.

    Returns:
        Figure with a 2 × 4 grid of axes in class id order.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for class_id, ax in enumerate(axes.flat):
        for split, color in SPLIT_COLORS.items():
            values = objects.filter(pl.col("split") == split, pl.col("class_id") == class_id)[
                column
            ].to_numpy()
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=split,
            )
        ax.set(xscale="log", title=CLASS_NAMES[class_id], yticks=[])
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    axes[0, 0].legend(title="Split")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
