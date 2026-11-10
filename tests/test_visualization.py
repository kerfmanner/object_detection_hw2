import re

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from src.data.classes import CLASS_NAMES
from src.training.experiment import REPO_ROOT
from src.visualization.style import (
    CLASS_COLORS,
    FAMILY_COLORS,
    GT_COLOR,
    IGNORED_REGION_COLOR,
    OCCLUSION_COLORS,
    PREDICTION_COLOR,
    SPLIT_COLORS,
    TEST_COLOR,
    set_plot_theme,
)


def test_class_colors_cover_all_classes_in_order() -> None:
    assert tuple(CLASS_COLORS) == CLASS_NAMES
    assert len(CLASS_NAMES) == 8


def test_all_colors_are_distinct_hex_values() -> None:
    colors = [
        *CLASS_COLORS.values(),
        *SPLIT_COLORS.values(),
        *OCCLUSION_COLORS.values(),
        TEST_COLOR,
        GT_COLOR,
        PREDICTION_COLOR,
        IGNORED_REGION_COLOR,
    ]
    assert all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in colors)
    assert len(set(colors)) == len(colors)


def test_family_colors_cover_every_config_family() -> None:
    families = {path.name for path in (REPO_ROOT / "config").iterdir() if path.is_dir()}
    assert set(FAMILY_COLORS) == families
    assert len(set(FAMILY_COLORS.values())) == len(FAMILY_COLORS)
    assert all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in FAMILY_COLORS.values())


def test_set_plot_theme_uses_class_palette() -> None:
    set_plot_theme()
    cycle = [to_hex(color) for color in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    assert cycle == list(CLASS_COLORS.values())
