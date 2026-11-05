import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.visualization.plots import plot_class_histograms


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
