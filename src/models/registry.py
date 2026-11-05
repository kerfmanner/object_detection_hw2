from collections.abc import Callable

import torch

from src.models.yolo import build_yolo

ModelBuilder = Callable[[dict], torch.nn.Module]

MODEL_BUILDERS: dict[str, ModelBuilder] = {"yolo": build_yolo}


def build_model(config: dict) -> torch.nn.Module:
    """Build the detector of a resolved experiment configuration.

    Args:
        config: Resolved configuration whose `run.family` names the detector family.

    Returns:
        The detector returned by the family's registered builder.
    """
    family = config["run"]["family"]
    if family not in MODEL_BUILDERS:
        raise ValueError(
            f"No model builder registered for family {family!r}; "
            f"registered families: {sorted(MODEL_BUILDERS) or 'none yet'}"
        )
    return MODEL_BUILDERS[family](config)
