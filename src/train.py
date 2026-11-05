import argparse
import logging
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from decouple import config

from src.models.registry import build_model
from src.training.experiment import (
    CONFIG_FILE,
    create_run_dir,
    format_toml,
    init_wandb,
    load_experiment_config,
)
from src.training.loop import train_detector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
WANDB_CONFIG_PATH = PROJECT_ROOT / "config" / "wandb.toml"


def run_training(
    config_path: Path, data_root: Path, runs_root: Path, is_smoke: bool
) -> Path | None:
    """Resolve an experiment config, prepare its run, and train the detector.

    The model is built before anything is written, so an unknown family leaves no run
    directory. A smoke run disables W&B and writes nothing to `runs_root`.

    Args:
        config_path: Path `config/<family>/<name>.toml` of the experiment.
        data_root: Dataset root with the provided splits.
        runs_root: Root directory of all runs, e.g. `runs/`.
        is_smoke: Whether to run the CPU smoke test instead of a full training.

    Returns:
        The run directory with the resolved `config.toml`, or None for a smoke run.
    """
    config = load_experiment_config(config_path)
    model = build_model(config)
    experiment = config["run"]["experiment"]
    run_dir = None if is_smoke else create_run_dir(runs_root, experiment, datetime.now())
    if run_dir is not None:
        (run_dir / CONFIG_FILE).write_text(format_toml(config))
    wandb_settings = tomllib.loads(WANDB_CONFIG_PATH.read_text())["wandb"]
    wandb_run = init_wandb(config, wandb_settings, run_dir)
    try:
        logger.info("Training %s, run directory %s", experiment, run_dir or "none (smoke)")
        train_detector(model, config, data_root, run_dir, wandb_run)
    finally:
        wandb_run.finish()
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Train a detector from an experiment config.")
    parser.add_argument("--config", type=Path, required=True, help="config/<family>/<name>.toml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One CPU training and evaluation step on the smoke subset; W&B disabled.",
    )
    args = parser.parse_args()
    run_training(args.config, Path(config("DATA_ROOT")), RUNS_ROOT, args.smoke)


if __name__ == "__main__":
    main()
