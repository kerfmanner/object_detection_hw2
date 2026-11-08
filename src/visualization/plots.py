import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.container import BarContainer
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


def plot_training_history(
    histories: pl.DataFrame,
    metrics: dict[str, str],
    title: str,
    x: str = "epoch",
    xlabel: str = "Epoch",
) -> Figure:
    """Plot per-epoch metrics of several runs, one panel per metric and one line per run.

    Args:
        histories: Long table with `experiment`, the `x` column, and one column per metric,
            e.g. from `load_run_histories`.
        metrics: Mapping from metric column to its axis label, in panel order.
        title: Figure title.
        x: Column of the shared x axis, e.g. `epoch` or cumulative training hours.
        xlabel: Label of the shared x axis, with units.

    Returns:
        Figure with one axes per metric in one row.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4), squeeze=False)
    experiments = histories["experiment"].unique(maintain_order=True).to_list()
    for ax, (column, label) in zip(axes[0], metrics.items(), strict=True):
        for experiment in experiments:
            run = histories.filter(pl.col("experiment") == experiment).sort(x)
            ax.plot(run[x], run[column], marker="o", markersize=4, label=experiment)
        ax.set(xlabel=xlabel, ylabel=label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_grouped_bars(
    table: pl.DataFrame,
    x: str,
    y: str,
    hue: str,
    labels: tuple[str, str, str],
    value_format: str,
) -> Figure:
    """Plot a bar per row of a long table, grouped along x and colored by hue.

    Args:
        table: Long table with the `x`, `y`, and `hue` columns, in the order to plot.
        x: Column of the groups on the x axis.
        y: Column of the bar heights.
        hue: Column that tells the bars of a group apart.
        labels: Label of the x axis, label of the y axis with units, and figure title.
        value_format: Format string of the value printed on each bar, e.g. `{:.3f}`.

    Returns:
        Figure with one axes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(table.to_dict(as_series=False), x=x, y=y, hue=hue, ax=ax)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt=value_format, fontsize=7)
    xlabel, ylabel, title = labels
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    return fig
