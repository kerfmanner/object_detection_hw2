import math

import torch
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNet

from src.data.classes import CLASS_NAMES
from src.models.anchors import build_anchor_generator

SUPPORTED_ARCH = "retinanet_resnet50_fpn_v2"
SUPPORTED_INIT = "coco"
CLASSIFIER_WEIGHT_STD = 0.01
PRIOR_PROBABILITY = 0.01


def build_classification_logits(
    in_channels: int, num_anchors: int, num_classes: int, seed: int
) -> torch.nn.Conv2d:
    """Build a new final classification convolution for RetinaNet, seeded.

    Shaped and initialized like torchvision's `RetinaNetClassificationHead`: a 3 x 3
    convolution with stride 1 and padding 1, normal weights with std 0.01, and the bias of
    a prior probability of 0.01 for every class, as in the focal loss paper
    (https://arxiv.org/abs/1708.02002;
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py).

    Args:
        in_channels: Channels of the classification tower output.
        num_anchors: Anchors per location.
        num_classes: Output classes per anchor, including torchvision's background column.
        seed: Seed of the weight initialization.

    Returns:
        Convolution with `num_anchors * num_classes` output channels.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        logits = torch.nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(logits.weight, std=CLASSIFIER_WEIGHT_STD)
    torch.nn.init.constant_(
        logits.get_parameter("bias"), -math.log((1 - PRIOR_PROBABILITY) / PRIOR_PROBABILITY)
    )
    return logits


def build_retinanet(config: dict) -> RetinaNet:
    """Build RetinaNet ResNet50-FPN v2 from COCO weights with a classifier for our classes.

    All COCO weights are loaded, including the classification tower; only the final
    classification convolution is replaced, with 9 columns per anchor (torchvision's unused
    background column 0, then the 8 classes), so labels follow the same +1 shift as Faster
    R-CNN. The box regression head keeps the v2 GIoU loss and the focal loss keeps
    torchvision's alpha 0.25 and gamma 2. The configured anchor generator replaces the
    default one and must keep 9 anchors per location, so that the COCO heads still match.

    Args:
        config: Resolved configuration with `arch`, `init`, `trainable_backbone_layers`,
            `min_size`, `max_size`, `seed`, and the `[anchors]`, `[matching]`, and
            `[inference]` tables of `config/retinanet/baseline.toml`.

    Returns:
        The torchvision detector with 9 output classes per anchor.
    """
    if config["arch"] != SUPPORTED_ARCH or config["init"] != SUPPORTED_INIT:
        raise ValueError(
            f"RetinaNet supports arch {SUPPORTED_ARCH!r} with init {SUPPORTED_INIT!r} only, "
            f"got {config['arch']!r} with {config['init']!r}"
        )
    anchors, matching, inference = config["anchors"], config["matching"], config["inference"]
    detector = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        trainable_backbone_layers=config["trainable_backbone_layers"],
        min_size=config["min_size"],
        max_size=config["max_size"],
        fg_iou_thresh=matching["fg_iou_thresh"],
        bg_iou_thresh=matching["bg_iou_thresh"],
        score_thresh=inference["score_thresh"],
        nms_thresh=inference["nms_thresh"],
        detections_per_img=inference["detections_per_img"],
        topk_candidates=inference["topk_candidates"],
    )
    detector.anchor_generator = build_anchor_generator(
        anchors["sizes"], anchors["aspect_ratios"], detector.anchor_generator
    )
    head = detector.head.classification_head
    num_classes = len(CLASS_NAMES) + 1
    head.cls_logits = build_classification_logits(
        head.cls_logits.in_channels, head.num_anchors, num_classes, config["seed"]
    )
    head.num_classes = num_classes
    return detector
