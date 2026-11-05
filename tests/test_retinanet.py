import math

import pytest
import torch
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNet

from src.models.retinanet import build_retinanet
from src.training.experiment import load_experiment_config
from tests.test_smoke import RETINANET_BASELINE

CLASSIFIER_KEY = "head.classification_head.cls_logits"


@pytest.fixture(scope="module")
def baseline_config() -> dict:
    return load_experiment_config(RETINANET_BASELINE)


@pytest.fixture(scope="module")
def baseline_model(baseline_config: dict) -> RetinaNet:
    return build_retinanet(baseline_config)


def test_baseline_keeps_torchvision_defaults_except_inference_and_classes(
    baseline_model: RetinaNet,
) -> None:
    default = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None)
    assert baseline_model.transform.min_size == (1024,)
    assert baseline_model.transform.max_size == 1824
    assert baseline_model.anchor_generator.sizes == default.anchor_generator.sizes
    assert baseline_model.anchor_generator.aspect_ratios == default.anchor_generator.aspect_ratios
    matcher, default_matcher = baseline_model.proposal_matcher, default.proposal_matcher
    assert (matcher.high_threshold, matcher.low_threshold) == (
        default_matcher.high_threshold,
        default_matcher.low_threshold,
    )
    assert matcher.allow_low_quality_matches == default_matcher.allow_low_quality_matches
    assert baseline_model.topk_candidates == default.topk_candidates
    assert (
        baseline_model.score_thresh,
        baseline_model.nms_thresh,
        baseline_model.detections_per_img,
    ) == (0.001, 0.5, 500)
    head = baseline_model.head.classification_head
    assert head.num_classes == 9 and head.cls_logits.out_channels == 9 * 9


def test_coco_weights_are_kept_and_the_new_classifier_is_seeded(
    baseline_config: dict, baseline_model: RetinaNet
) -> None:
    coco = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict()
    state = baseline_model.state_dict()
    for key in (
        "head.classification_head.conv.0.0.weight",
        "head.regression_head.bbox_reg.weight",
        "backbone.fpn.layer_blocks.0.0.weight",
    ):
        assert torch.equal(state[key], coco[key])
    weight_key = f"{CLASSIFIER_KEY}.weight"
    assert state[weight_key].std().item() == pytest.approx(0.01, rel=0.05)
    prior_bias = -math.log((1 - 0.01) / 0.01)
    assert torch.allclose(state[f"{CLASSIFIER_KEY}.bias"], torch.full((81,), prior_bias))
    again = build_retinanet(baseline_config).state_dict()[weight_key]
    other_seed = build_retinanet(baseline_config | {"seed": 7}).state_dict()[weight_key]
    assert torch.equal(again, state[weight_key])
    assert not torch.equal(other_seed, state[weight_key])


def test_anchors_that_change_the_head_shape_are_rejected(baseline_config: dict) -> None:
    anchors = baseline_config["anchors"] | {"aspect_ratios": [[0.5, 1.0, 1.5, 2.0]] * 5}
    with pytest.raises(ValueError, match="Anchors per location"):
        build_retinanet(baseline_config | {"anchors": anchors})
