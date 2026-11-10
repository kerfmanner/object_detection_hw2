import matplotlib.pyplot as plt
import seaborn as sns

from src.data.classes import CLASS_NAMES

# Hex values of the seaborn "colorblind" palette:
# https://seaborn.pydata.org/tutorial/color_palettes.html
CLASS_COLORS = dict(
    zip(
        CLASS_NAMES,
        ("#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494"),
        strict=True,
    )
)
GT_COLOR = "#ece133"
PREDICTION_COLOR = "#56b4e9"
IGNORED_REGION_COLOR = "#000000"
IGNORED_REGION_ALPHA = 0.45


def set_plot_theme() -> None:
    """Apply the shared seaborn theme and class palette to all following matplotlib figures."""
    sns.set_theme(context="notebook", style="whitegrid", palette=list(CLASS_COLORS.values()))
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 100, "savefig.bbox": "tight"})


SPLIT_COLORS = {"train": "#3a5f9e", "val": "#d9822b"}
TEST_COLOR = "#4c9a6a"
OCCLUSION_COLORS = {"none": "#c6dbef", "partial": "#6baed6", "heavy": "#08519c"}
ERROR_CATEGORY_COLORS = {
    "correct": "#029e73",
    "duplicate": "#ece133",
    "wrong_class": "#d55e00",
    "poor_localisation": "#cc78bc",
    "background": "#949494",
}
SIZE_BUCKET_COLORS = {"small": "#9ecae1", "medium": "#4292c6", "large": "#08306b"}
FEATURE_LEVEL_COLORS = ("#e66101", "#5e3c99", "#1a9850")
FAMILY_COLORS = {
    "faster_rcnn": "#1b9e77",
    "retinanet": "#d95f02",
    "yolo": "#7570b3",
    "rtdetr": "#e7298a",
    "postprocess": "#444444",
}
