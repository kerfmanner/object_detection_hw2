import torch

SUPPORTED_SCHEDULE = "constant"
BACKBONE_PREFIX = "detector.model.backbone."
NORM_LAYERS = (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)


def build_rtdetr_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.AdamW:
    """Build AdamW with the parameter groups of the official RT-DETRv2 recipe.

    Backbone parameters train at `backbone_lr`, normalization layers outside the backbone
    without weight decay (`norm_weight_decay`), and all other parameters at `lr` with
    `weight_decay`, as in
    https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetrv2/include/optimizer.yml
    Frozen parameters are left out. The first parameter group is the encoder and decoder
    group, whose LR is the one logged. Warmup is applied by the training loop.

    Args:
        model: `RTDetrDetector`, whose backbone parameters are named `detector.model.backbone.*`
            and whose frozen parameters have `requires_grad` False.
        config: Resolved configuration with `lr`, `backbone_lr`, `weight_decay`,
            `norm_weight_decay`, `betas`, `schedule`, and `warmup_iters`.

    Returns:
        The optimizer.
    """
    warmup_iters = config["warmup_iters"]
    if config["schedule"] != SUPPORTED_SCHEDULE:
        raise ValueError(
            f"RT-DETR supports schedule {SUPPORTED_SCHEDULE!r} only, got {config['schedule']!r}"
        )
    if type(warmup_iters) is not int or warmup_iters < 0:
        raise ValueError(f"warmup_iters must be an integer of at least 0, got {warmup_iters}")
    norm_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, NORM_LAYERS)
        for p in module.parameters(recurse=False)
    }
    groups: dict[str, list[torch.nn.Parameter]] = {"main": [], "norm": [], "backbone": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith(BACKBONE_PREFIX)
        group = "backbone" if is_backbone else "norm" if id(parameter) in norm_ids else "main"
        groups[group].append(parameter)
    settings = {
        "main": {"lr": config["lr"], "weight_decay": config["weight_decay"]},
        "norm": {"lr": config["lr"], "weight_decay": config["norm_weight_decay"]},
        "backbone": {"lr": config["backbone_lr"], "weight_decay": config["weight_decay"]},
    }
    return torch.optim.AdamW(
        [{"params": params, **settings[name]} for name, params in groups.items() if params],
        lr=config["lr"],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )
