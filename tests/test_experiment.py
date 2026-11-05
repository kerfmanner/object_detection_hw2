import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from src.models.registry import build_model
from src.training.experiment import (
    create_run_dir,
    format_toml,
    init_wandb,
    load_experiment_config,
    merge_config,
    name_experiment,
)

BASELINE_TOML = """# Faster R-CNN baseline
arch = "fasterrcnn_resnet50_fpn_v2"
seed = 42
lr = 0.01
amp = true

[rpn]
post_nms_top_n_train = 2000
anchor_sizes = [[32], [64], [128], [256], [512]]

[inference]
score_threshold = 0.05
"""


@pytest.fixture
def family_dir(tmp_path: Path) -> Path:
    family_dir = tmp_path / "config" / "faster_rcnn"
    family_dir.mkdir(parents=True)
    (family_dir / "baseline.toml").write_text(BASELINE_TOML)
    return family_dir


def test_merge_replaces_nested_values_and_keeps_the_baseline() -> None:
    baseline = {"lr": 0.01, "rpn": {"top_n": 2000, "sizes": [32]}, "seed": 42}
    merged = merge_config(baseline, {"rpn": {"top_n": 1000}, "lr": 0.02})
    assert merged == {"lr": 0.02, "rpn": {"top_n": 1000, "sizes": [32]}, "seed": 42}
    assert baseline["rpn"]["top_n"] == 2000


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"learning_rate": 0.1}, "'learning_rate', which the baseline does not define"),
        ({"rpn": {"topn": 5}}, "'rpn.topn', which the baseline does not define"),
        ({"rpn": 5}, "whether 'rpn' is a table"),
        ({"lr": {"value": 1}}, "whether 'lr' is a table"),
    ],
)
def test_merge_rejects_keys_and_tables_the_baseline_lacks(override: dict, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        merge_config({"lr": 0.01, "rpn": {"top_n": 2000}}, override)


def test_experiment_names_follow_the_config_file() -> None:
    assert name_experiment("faster_rcnn", "baseline", 42, 42) == "faster_rcnn-baseline"
    assert (
        name_experiment("faster_rcnn", "rpn_post_nms_2000", 7, 42)
        == "faster_rcnn-rpn_post_nms_2000-s7"
    )


def test_load_baseline_and_override_configs(family_dir: Path) -> None:
    baseline = load_experiment_config(family_dir / "baseline.toml")
    assert baseline["run"] == {
        "experiment": "faster_rcnn-baseline",
        "family": "faster_rcnn",
        "config": (family_dir / "baseline.toml").as_posix(),
        "baseline": (family_dir / "baseline.toml").as_posix(),
    }
    override_path = family_dir / "rpn_post_nms_1000.toml"
    override_path.write_text("# Fewer RPN proposals\n[rpn]\npost_nms_top_n_train = 1000\n")
    config = load_experiment_config(override_path)
    assert config["rpn"]["post_nms_top_n_train"] == 1000
    assert config["rpn"]["anchor_sizes"] == baseline["rpn"]["anchor_sizes"]
    assert config["run"]["experiment"] == "faster_rcnn-rpn_post_nms_1000"
    seed_path = family_dir / "rpn_post_nms_1000_seed.toml"
    seed_path.write_text("[rpn]\npost_nms_top_n_train = 1000\nseed = 1\n")
    with pytest.raises(ValueError, match="'rpn.seed'"):
        load_experiment_config(seed_path)
    seed_path.write_text("seed = 1\n[rpn]\npost_nms_top_n_train = 1000\n")
    assert load_experiment_config(seed_path)["run"]["experiment"].endswith("_seed-s1")


def test_baseline_without_seed_is_rejected(family_dir: Path) -> None:
    (family_dir / "baseline.toml").write_text('arch = "x"\n')
    with pytest.raises(ValueError, match="does not set `seed`"):
        load_experiment_config(family_dir / "baseline.toml")


def test_format_toml_round_trips_resolved_configs(family_dir: Path) -> None:
    config = load_experiment_config(family_dir / "baseline.toml") | {
        "text": 'quote " and ünïcode',
        "small": 1e-5,
        "empty": [],
        "deep": {"a": {"b": {"c": [1.5, 2]}}, "flag": False},
    }
    text = format_toml(config)
    assert tomllib.loads(text) == config
    assert text.index("seed = 42") < text.index("[rpn]")


def test_create_run_dir_uses_the_timestamp_and_never_reuses(tmp_path: Path) -> None:
    started_at = datetime(2026, 11, 6, 9, 5)
    run_dir = create_run_dir(tmp_path / "runs", "faster_rcnn-baseline", started_at)
    assert run_dir == tmp_path / "runs" / "faster_rcnn-baseline" / "20261106-0905"
    assert run_dir.is_dir() and not any(run_dir.iterdir())
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path / "runs", "faster_rcnn-baseline", started_at)


def test_smoke_wandb_run_is_disabled_and_named_after_the_experiment(family_dir: Path) -> None:
    config = load_experiment_config(family_dir / "baseline.toml")
    run = init_wandb(config, {"entity": "entity", "project": "project"}, run_dir=None)
    assert run.name == "faster_rcnn-baseline"
    assert run.disabled
    run.finish()


def test_unknown_family_raises_a_clear_error(family_dir: Path) -> None:
    config = load_experiment_config(family_dir / "baseline.toml")
    with pytest.raises(ValueError, match="No model builder registered for family 'faster_rcnn'"):
        build_model(config)
