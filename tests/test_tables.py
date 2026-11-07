import json
from pathlib import Path

import polars as pl
import pytest

from src.evaluate import METRICS_FILE
from src.visualization.tables import compare_with_reference, load_run_histories


def test_run_histories_are_stacked_with_the_experiment_name(tmp_path: Path) -> None:
    run_dirs = {}
    for experiment, history in {
        "a": [{"epoch": 1, "val/map": 0.1}, {"epoch": 2, "val/map": 0.2}],
        "b": [{"epoch": 1, "val/map": 0.3, "train/loss": 1.0}],
    }.items():
        run_dirs[experiment] = tmp_path / experiment
        run_dirs[experiment].mkdir()
        metrics = {"training": {"history": history}}
        (run_dirs[experiment] / METRICS_FILE).write_text(json.dumps(metrics))
    histories = load_run_histories(run_dirs)
    assert histories.columns[:2] == ["experiment", "epoch"]
    assert histories.select("experiment", "epoch", "val/map").rows() == [
        ("a", 1, 0.1),
        ("a", 2, 0.2),
        ("b", 1, 0.3),
    ]
    assert histories["train/loss"].to_list() == [None, None, 1.0]


def test_comparison_subtracts_the_reference_row() -> None:
    results = pl.DataFrame(
        {"experiment": ["base", "new"], "map": [0.3, 0.32], "ap_small": [0.1, 0.09]}
    )
    compared = compare_with_reference(results, "base", ["map", "ap_small"])
    assert compared.columns == ["experiment", "map", "ap_small", "delta_map", "delta_ap_small"]
    assert compared["delta_map"].to_list() == pytest.approx([0.0, 0.02])
    assert compared["delta_ap_small"].to_list() == pytest.approx([0.0, -0.01])
    with pytest.raises(ValueError, match="Expected one results row"):
        compare_with_reference(results, "missing", ["map"])
