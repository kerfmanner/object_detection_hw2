import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

from src.visualization.plots import (
    plot_class_histograms,
    plot_embedding_panels,
    plot_grouped_bars,
    plot_training_history,
)


def test_plot_class_histograms_draws_one_panel_per_class() -> None:
    objects = pl.DataFrame(
        {
            "split": ["train"] * 8 + ["val"] * 8,
            "class_id": list(range(8)) * 2,
            "size": [10.0] * 16,
        }
    )
    fig = plot_class_histograms(objects, "size", np.geomspace(1, 100, 10), "Size (px)", "Sizes")
    assert len(fig.axes) == 8
    assert [ax.get_title() for ax in fig.axes][:2] == ["person", "bicycle"]
    plt.close(fig)


def test_training_history_draws_one_line_per_run_and_panel_per_metric() -> None:
    histories = pl.DataFrame(
        {
            "experiment": ["a", "a", "b"],
            "epoch": [1, 2, 1],
            "val/map": [0.1, 0.2, 0.15],
            "train/loss": [1.0, 0.8, 0.9],
        }
    )
    fig = plot_training_history(histories, {"val/map": "mAP", "train/loss": "Loss"}, "History")
    assert [ax.get_ylabel() for ax in fig.axes] == ["mAP", "Loss"]
    assert [len(ax.lines) for ax in fig.axes] == [2, 2]
    assert np.asarray(fig.axes[0].lines[0].get_ydata()).tolist() == [0.1, 0.2]
    plt.close(fig)


def test_grouped_bars_draw_one_bar_per_row() -> None:
    table = pl.DataFrame(
        {"size": ["small", "small", "large"], "recall": [0.2, 0.3, 0.9], "model": ["a", "b", "a"]}
    )
    fig = plot_grouped_bars(
        table, "size", "recall", "model", ("Size", "Recall", "Recall"), "{:.2f}"
    )
    bars = [patch for patch in fig.axes[0].patches if isinstance(patch, Rectangle)]
    heights = sorted(bar.get_height() for bar in bars if bar.get_height())
    assert heights == [0.2, 0.3, 0.9]
    assert sorted(text.get_text() for text in fig.axes[0].texts) == ["0.20", "0.30", "0.90"]
    plt.close(fig)


def test_training_history_uses_the_given_x_column() -> None:
    histories = pl.DataFrame(
        {"experiment": ["a", "a"], "epoch": [1, 2], "hours": [0.5, 1.0], "m": [0.1, 0.2]}
    )
    fig = plot_training_history(histories, {"m": "mAP"}, "History", x="hours", xlabel="Hours")
    assert np.asarray(fig.axes[0].lines[0].get_xdata()).tolist() == [0.5, 1.0]
    assert fig.axes[0].get_xlabel() == "Hours"
    plt.close(fig)


def test_embedding_panels_draw_only_the_labels_of_their_palette() -> None:
    table = pl.DataFrame(
        {
            "embedding_x": [0.0, 1.0, 2.0, 3.0],
            "embedding_y": [0.0, 1.0, 0.0, 1.0],
            "category": ["correct", "background", "ignored", "correct"],
            "size": ["small", "small", "large", "large"],
        }
    )
    fig = plot_embedding_panels(
        table,
        [
            ("category", {"correct": "#00ff00", "background": "#999999"}, "By category"),
            ("size", {"small": "#0000ff", "large": "#000000"}, "By size"),
        ],
        "embeddings",
    )
    counts = [sum(len(np.asarray(c.get_offsets())) for c in ax.collections) for ax in fig.axes]
    assert counts == [3, 4]
    assert [ax.get_title() for ax in fig.axes] == ["By category", "By size"]
    plt.close(fig)
