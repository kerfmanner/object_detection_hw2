import json
import tomllib
from datetime import datetime
from pathlib import Path

import wandb

BASELINE_STEM = "baseline"
RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M"
CONFIG_FILE = "config.toml"
TomlValue = str | int | float | bool | list | dict


def merge_config(baseline: dict, override: dict, path: str = "") -> dict:
    """Merge experiment overrides into a family baseline, table by table.

    Args:
        baseline: Baseline configuration that sets every supported option.
        override: Experiment configuration with the options it changes.
        path: Dotted key of the tables being merged, used in error messages.

    Returns:
        A new configuration with the override values replacing the baseline values.
    """
    merged = dict(baseline)
    for key, value in override.items():
        dotted = f"{path}{key}"
        if key not in baseline:
            raise ValueError(f"Override sets {dotted!r}, which the baseline does not define")
        if isinstance(baseline[key], dict) != isinstance(value, dict):
            raise ValueError(f"Override changes whether {dotted!r} is a table")
        is_table = isinstance(value, dict)
        merged[key] = merge_config(baseline[key], value, f"{dotted}.") if is_table else value
    return merged


def name_experiment(family: str, config_stem: str, seed: int, baseline_seed: int) -> str:
    """Name an experiment `<family>-<config_stem>`, with `-s<seed>` for seed repeats.

    Args:
        family: Detector family, the name of the config subdirectory.
        config_stem: File name of the experiment config without `.toml`.
        seed: Seed of the resolved configuration.
        baseline_seed: Seed of the family baseline, which gets no suffix.

    Returns:
        The experiment name used for the run directory, W&B, and `results.csv`.
    """
    suffix = "" if seed == baseline_seed else f"-s{seed}"
    return f"{family}-{config_stem}{suffix}"


def load_experiment_config(config_path: Path) -> dict:
    """Load `config/<family>/<name>.toml` merged over `config/<family>/baseline.toml`.

    Args:
        config_path: Path of the baseline or of an experiment config of one family.

    Returns:
        Resolved configuration with a `run` table holding `experiment`, `family`, `config`
        (path of `config_path`), and `baseline` (path of the baseline file).
    """
    family = config_path.parent.name
    baseline_path = config_path.parent / f"{BASELINE_STEM}.toml"
    baseline = tomllib.loads(baseline_path.read_text())
    if "seed" not in baseline:
        raise ValueError(f"{baseline_path} does not set `seed`")
    override = {} if config_path == baseline_path else tomllib.loads(config_path.read_text())
    config = merge_config(baseline, override)
    experiment = name_experiment(family, config_path.stem, config["seed"], baseline["seed"])
    return config | {
        "run": {
            "experiment": experiment,
            "family": family,
            "config": config_path.as_posix(),
            "baseline": baseline_path.as_posix(),
        }
    }


def format_toml_value(value: TomlValue) -> str:
    """Format a scalar or array as a TOML value.

    JSON strings, numbers, booleans, and arrays of them are valid TOML values.

    Args:
        value: String, integer, finite float, boolean, or (nested) list of them.

    Returns:
        The TOML representation.
    """
    return json.dumps(value, allow_nan=False)


def format_toml(config: dict, table: str = "") -> str:
    """Format a configuration dictionary as a TOML document.

    Args:
        config: Mapping of keys to TOML values or nested tables.
        table: Dotted name of the table being formatted, empty at the top level.

    Returns:
        TOML text with the scalar keys of each table before its subtables.
    """
    lines = [f"[{table}]"] if table else []
    lines += [
        f"{key} = {format_toml_value(value)}"
        for key, value in config.items()
        if not isinstance(value, dict)
    ]
    sections = ["\n".join(lines)] if lines else []
    sections += [
        format_toml(value, f"{table}.{key}" if table else key).rstrip("\n")
        for key, value in config.items()
        if isinstance(value, dict)
    ]
    return "\n\n".join(sections) + "\n"


def create_run_dir(runs_root: Path, experiment: str, started_at: datetime) -> Path:
    """Create the run directory `<runs_root>/<experiment>/<YYYYMMDD-HHMM>/`.

    Args:
        runs_root: Root directory of all runs, e.g. `runs/`.
        experiment: Experiment name from `name_experiment`.
        started_at: Start time of the run.

    Returns:
        The new, empty run directory; an existing directory is never reused.
    """
    run_dir = runs_root / experiment / started_at.strftime(RUN_TIMESTAMP_FORMAT)
    run_dir.mkdir(parents=True)
    return run_dir


def init_wandb(config: dict, wandb_settings: dict, run_dir: Path | None) -> wandb.Run:
    """Start the W&B run of an experiment; smoke runs pass no run directory and are disabled.

    Args:
        config: Resolved configuration from `load_experiment_config`.
        wandb_settings: The `[wandb]` table of `config/wandb.toml` with `entity` and
            `project`.
        run_dir: Run directory for W&B files, or None for a disabled smoke run.

    Returns:
        The W&B run named after the experiment, with the full configuration.
    """
    return wandb.init(
        entity=wandb_settings["entity"],
        project=wandb_settings["project"],
        name=config["run"]["experiment"],
        config=config,
        dir=run_dir,
        mode="disabled" if run_dir is None else None,
    )
