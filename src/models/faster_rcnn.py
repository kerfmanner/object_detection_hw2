import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

from src.data.classes import CLASS_NAMES
from src.models.anchors import build_anchor_generator

SUPPORTED_ARCH = "fasterrcnn_resnet50_fpn_v2"
SUPPORTED_INIT = "coco"


def build_faster_rcnn(config: dict) -> FasterRCNN:
    """Build Faster R-CNN ResNet50-FPN v2 from COCO weights with a box predictor for our classes.

    All COCO weights are loaded, then the box predictor is replaced by one for the 8 classes
    plus background, as in the torchvision fine-tuning tutorial
    (https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). The new
    predictor is initialized from the config seed. The configured anchor generator replaces
    the default one; it must keep the number of anchors per location, so that the COCO RPN
    head still matches it. Labels stay torchvision's, with background 0; the training loop
    and inference shift them with `src.models.torchvision_labels`.

    Args:
        config: Resolved configuration with `arch`, `init`, `trainable_backbone_layers`,
            `min_size`, `max_size`, `seed`, and the `[rpn]`, `[roi_heads]`, and `[inference]`
            tables of `config/faster_rcnn/baseline.toml`.

    Returns:
        The torchvision detector with 9 output classes, background first.
    """
    if config["arch"] != SUPPORTED_ARCH or config["init"] != SUPPORTED_INIT:
        raise ValueError(
            f"Faster R-CNN supports arch {SUPPORTED_ARCH!r} with init {SUPPORTED_INIT!r} only, "
            f"got {config['arch']!r} with {config['init']!r}"
        )
    rpn, roi_heads, inference = config["rpn"], config["roi_heads"], config["inference"]
    detector = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        trainable_backbone_layers=config["trainable_backbone_layers"],
        min_size=config["min_size"],
        max_size=config["max_size"],
        rpn_pre_nms_top_n_train=rpn["pre_nms_top_n_train"],
        rpn_pre_nms_top_n_test=rpn["pre_nms_top_n_test"],
        rpn_post_nms_top_n_train=rpn["post_nms_top_n_train"],
        rpn_post_nms_top_n_test=rpn["post_nms_top_n_test"],
        rpn_nms_thresh=rpn["nms_thresh"],
        rpn_fg_iou_thresh=rpn["fg_iou_thresh"],
        rpn_bg_iou_thresh=rpn["bg_iou_thresh"],
        rpn_batch_size_per_image=rpn["batch_size_per_image"],
        rpn_positive_fraction=rpn["positive_fraction"],
        rpn_score_thresh=rpn["score_thresh"],
        box_score_thresh=inference["score_thresh"],
        box_nms_thresh=inference["nms_thresh"],
        box_detections_per_img=inference["detections_per_img"],
        box_fg_iou_thresh=roi_heads["fg_iou_thresh"],
        box_bg_iou_thresh=roi_heads["bg_iou_thresh"],
        box_batch_size_per_image=roi_heads["batch_size_per_image"],
        box_positive_fraction=roi_heads["positive_fraction"],
    )
    detector.rpn.anchor_generator = build_anchor_generator(
        rpn["anchor_sizes"], rpn["aspect_ratios"], detector.rpn.anchor_generator
    )
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    return detector
