import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import generalized_box_iou_loss

from src.data.classes import CLASS_NAMES
from src.models.anchors import build_anchor_generator

SUPPORTED_ARCH = "fasterrcnn_resnet50_fpn_v2"
SUPPORTED_INIT = "coco"
BOX_LOSSES = ("smooth_l1", "giou")
UNIT_BOX = (0.0, 0.0, 1.0, 1.0)


def build_faster_rcnn(config: dict) -> FasterRCNN:
    """Build Faster R-CNN ResNet50-FPN v2 from COCO weights with a box predictor for our classes.

    All COCO weights are loaded, then the box predictor is replaced by one for the 8 classes
    plus background, as in the torchvision fine-tuning tutorial
    (https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). The new
    predictor is initialized from the config seed. The configured anchor generator replaces
    the default one; it must keep the number of anchors per location, so that the COCO RPN
    head still matches it. Labels stay torchvision's, with background 0; the training loop
    and inference shift them with `src.models.torchvision_labels`. The box regression loss
    of `[roi_heads]` is not read here: it changes training only, so the training loop sets
    it (`set_box_regression_loss`) and prediction builds the same model without it.

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


def compute_giou_box_loss(
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
    box_coder: BoxCoder,
) -> torch.Tensor:
    """Compute the GIoU loss of the second-stage boxes, normalized like torchvision's loss.

    The GIoU loss (https://arxiv.org/abs/1902.09630) replaces the Smooth L1 loss of
    torchvision's `fastrcnn_loss`: it is summed over the positive RoIs, each with the box of
    its GT class, and divided by the number of sampled RoIs, as in mmdetection's Faster R-CNN
    with `reg_decoded_bbox`
    (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/bbox_head.py).
    Predictions and targets are decoded relative to the unit box instead of their proposal:
    decoding relative to a proposal maps those boxes by an axis-aligned scaling with positive
    factors and a shift, which leaves every area ratio and thus the GIoU unchanged.

    Args:
        box_regression: Tensor (RoIs, classes * 4) of encoded box offsets per class.
        labels: Per frame, tensor (RoIs,) of sampled RoI labels, background 0.
        regression_targets: Per frame, tensor (RoIs, 4) of encoded GT box offsets.
        box_coder: Box coder of the RoI heads, with their offset weights.

    Returns:
        Scalar loss tensor.
    """
    labels_all, targets_all = torch.cat(labels), torch.cat(regression_targets)
    positive = torch.where(labels_all > 0)[0]
    codes = box_regression.reshape(labels_all.numel(), -1, 4)[positive, labels_all[positive]]
    unit_boxes = torch.tensor(UNIT_BOX, device=codes.device).expand(positive.numel(), 4)
    predicted = box_coder.decode_single(codes.float(), unit_boxes)
    target = box_coder.decode_single(targets_all[positive].float(), unit_boxes)
    return generalized_box_iou_loss(predicted, target, reduction="sum") / labels_all.numel()


class GIoURoIHeads(RoIHeads):
    """torchvision box RoI heads trained with a weighted GIoU box regression loss.

    Training samples RoIs and computes the classification loss as torchvision's `RoIHeads`;
    the box loss is `giou_loss_weight` times `compute_giou_box_loss`, under the torchvision
    name `loss_box_reg`. Inference is torchvision's. The heads have the same parameters as
    the ones they replace, so checkpoints load into either.

    Args:
        roi_heads: torchvision box RoI heads whose modules and settings are reused.
        giou_loss_weight: Factor of the GIoU loss.

    Attributes:
        giou_loss_weight: Factor of the GIoU loss.
    """

    def __init__(self, roi_heads: RoIHeads, giou_loss_weight: float) -> None:
        super().__init__(
            box_roi_pool=roi_heads.box_roi_pool,
            box_head=roi_heads.box_head,
            box_predictor=roi_heads.box_predictor,
            fg_iou_thresh=roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=roi_heads.box_coder.weights,
            score_thresh=roi_heads.score_thresh,
            nms_thresh=roi_heads.nms_thresh,
            detections_per_img=roi_heads.detections_per_img,
        )
        self.giou_loss_weight = giou_loss_weight

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        box_loss = compute_giou_box_loss(box_regression, labels, regression_targets, self.box_coder)
        losses = {
            "loss_classifier": F.cross_entropy(class_logits, torch.cat(labels)),
            "loss_box_reg": self.giou_loss_weight * box_loss,
        }
        return [], losses


def set_box_regression_loss(detector: FasterRCNN, roi_heads_config: dict) -> None:
    """Set the second-stage box regression loss of a Faster R-CNN before training.

    `smooth_l1` keeps torchvision's RoI heads and loss; `giou` replaces the heads by
    `GIoURoIHeads` with the same modules and settings.

    Args:
        detector: Detector from `build_faster_rcnn`.
        roi_heads_config: `[roi_heads]` table with `box_loss` (`smooth_l1` or `giou`) and
            `giou_loss_weight`.
    """
    box_loss, giou_loss_weight = roi_heads_config["box_loss"], roi_heads_config["giou_loss_weight"]
    if box_loss not in BOX_LOSSES:
        raise ValueError(f"box_loss must be one of {BOX_LOSSES}, got {box_loss!r}")
    if box_loss == "smooth_l1":
        return
    roi_heads = detector.roi_heads
    if not isinstance(roi_heads, RoIHeads):
        raise TypeError(f"Expected torchvision RoIHeads, got {type(roi_heads).__name__}")
    detector.roi_heads = GIoURoIHeads(roi_heads, giou_loss_weight)
