import pytest
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork

from src.models.faster_rcnn import build_faster_rcnn
from src.training.experiment import load_experiment_config
from tests.test_smoke import FASTER_RCNN_BASELINE


@pytest.fixture(scope="module")
def baseline_config() -> dict:
    return load_experiment_config(FASTER_RCNN_BASELINE)


@pytest.fixture(scope="module")
def baseline_model(baseline_config: dict) -> FasterRCNN:
    return build_faster_rcnn(baseline_config)


def test_baseline_keeps_torchvision_defaults_except_inference_and_classes(
    baseline_model: FasterRCNN,
) -> None:
    default = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    assert baseline_model.transform.min_size == (1024,)
    assert baseline_model.transform.max_size == 1824
    rpn, default_rpn = baseline_model.rpn, default.rpn
    heads, default_heads = baseline_model.roi_heads, default.roi_heads
    assert isinstance(rpn, RegionProposalNetwork) and isinstance(default_rpn, RegionProposalNetwork)
    assert isinstance(heads, RoIHeads) and isinstance(default_heads, RoIHeads)
    assert rpn.anchor_generator.sizes == default_rpn.anchor_generator.sizes
    assert rpn.anchor_generator.aspect_ratios == default_rpn.anchor_generator.aspect_ratios
    assert rpn._pre_nms_top_n == default_rpn._pre_nms_top_n
    assert rpn._post_nms_top_n == default_rpn._post_nms_top_n
    for attribute in ("nms_thresh", "score_thresh"):
        assert getattr(rpn, attribute) == getattr(default_rpn, attribute)
    for module, default_module in ((rpn, default_rpn), (heads, default_heads)):
        matcher, default_matcher = module.proposal_matcher, default_module.proposal_matcher
        assert (matcher.high_threshold, matcher.low_threshold) == (
            default_matcher.high_threshold,
            default_matcher.low_threshold,
        )
        sampler, default_sampler = module.fg_bg_sampler, default_module.fg_bg_sampler
        assert (sampler.batch_size_per_image, sampler.positive_fraction) == (
            default_sampler.batch_size_per_image,
            default_sampler.positive_fraction,
        )
    assert (heads.score_thresh, heads.nms_thresh, heads.detections_per_img) == (0.001, 0.5, 500)
    assert heads.box_predictor.cls_score.out_features == 9
    assert heads.box_predictor.bbox_pred.out_features == 36


def test_coco_weights_are_loaded_and_the_new_predictor_is_seeded(
    baseline_config: dict, baseline_model: FasterRCNN
) -> None:
    coco = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict()
    state = baseline_model.state_dict()
    for key in (
        "rpn.head.conv.0.0.weight",
        "roi_heads.box_head.5.weight",
        "backbone.fpn.layer_blocks.0.0.weight",
    ):
        assert torch.equal(state[key], coco[key])
    predictor_key = "roi_heads.box_predictor.cls_score.weight"
    again = build_faster_rcnn(baseline_config).state_dict()[predictor_key]
    other_seed = build_faster_rcnn(baseline_config | {"seed": 7}).state_dict()[predictor_key]
    assert torch.equal(again, state[predictor_key])
    assert not torch.equal(other_seed, state[predictor_key])


def test_anchors_that_change_the_rpn_head_shape_are_rejected(baseline_config: dict) -> None:
    rpn = baseline_config["rpn"] | {"aspect_ratios": [[0.5, 1.0, 1.5, 2.0]] * 5}
    with pytest.raises(ValueError, match="Anchors per location"):
        build_faster_rcnn(baseline_config | {"rpn": rpn})
