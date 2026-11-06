import copy

import pytest
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import generalized_box_iou_loss

from src.models.faster_rcnn import (
    GIoURoIHeads,
    build_faster_rcnn,
    compute_giou_box_loss,
    set_box_regression_loss,
)
from src.training.experiment import load_experiment_config
from tests.test_smoke import FASTER_RCNN_BASELINE

TINY_INPUT = {"min_size": 64, "max_size": 96}


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


def test_giou_box_loss_equals_the_loss_of_boxes_decoded_relative_to_their_proposals(
    baseline_model: FasterRCNN,
) -> None:
    heads = baseline_model.roi_heads
    assert isinstance(heads, RoIHeads)
    coder = heads.box_coder
    generator = torch.Generator().manual_seed(0)
    proposals = torch.tensor([[10.0, 20.0, 40.0, 30.0], [5.0, 5.0, 9.0, 25.0], [0, 0, 8, 8]])
    gt_boxes = proposals + torch.tensor([[2.0, -1.0, 5.0, 2.0], [1.0, 2.0, -1.0, 4.0], [0] * 4])
    labels = torch.tensor([3, 7, 0])
    targets = coder.encode_single(gt_boxes, proposals)
    box_regression = 0.5 * torch.randn(3, 9 * 4, generator=generator)
    loss = compute_giou_box_loss(
        box_regression, [labels[:2], labels[2:]], [targets[:2], targets[2:]], coder
    )
    codes = box_regression.reshape(3, 9, 4)[[0, 1], labels[:2]]
    decoded = coder.decode_single(codes, proposals[:2])
    expected = generalized_box_iou_loss(decoded, gt_boxes[:2], reduction="sum") / 3
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    perfect = torch.zeros(3, 9, 4)
    perfect[[0, 1], labels[:2]] = targets[:2]
    zero = compute_giou_box_loss(perfect.reshape(3, 36), [labels], [targets], coder)
    assert zero.item() == pytest.approx(0.0, abs=1e-6)


def test_giou_heads_keep_the_settings_parameters_and_inference_of_torchvision(
    baseline_config: dict,
) -> None:
    roi_heads: dict = baseline_config["roi_heads"]
    plain = build_faster_rcnn(baseline_config | TINY_INPUT)
    plain_heads = plain.roi_heads
    set_box_regression_loss(plain, roi_heads)
    assert plain.roi_heads is plain_heads
    giou = copy.deepcopy(plain)
    set_box_regression_loss(giou, roi_heads | {"box_loss": "giou"})
    heads = giou.roi_heads
    assert isinstance(heads, GIoURoIHeads) and isinstance(plain_heads, RoIHeads)
    assert heads.giou_loss_weight == 10.0
    assert heads.box_coder.weights == plain_heads.box_coder.weights
    assert (heads.proposal_matcher.high_threshold, heads.proposal_matcher.low_threshold) == (
        plain_heads.proposal_matcher.high_threshold,
        plain_heads.proposal_matcher.low_threshold,
    )
    assert (heads.fg_bg_sampler.batch_size_per_image, heads.fg_bg_sampler.positive_fraction) == (
        plain_heads.fg_bg_sampler.batch_size_per_image,
        plain_heads.fg_bg_sampler.positive_fraction,
    )
    assert (heads.score_thresh, heads.nms_thresh, heads.detections_per_img) == (0.001, 0.5, 500)
    plain_state, giou_state = plain.state_dict(), giou.state_dict()
    assert plain_state.keys() == giou_state.keys()
    assert all(torch.equal(plain_state[key], giou_state[key]) for key in plain_state)
    frames = [torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(1))]
    with torch.inference_mode():
        [plain_output] = plain.eval()(frames)
        [giou_output] = giou.eval()(frames)
    assert all(torch.equal(plain_output[key], giou_output[key]) for key in plain_output)


def test_giou_heads_train_with_the_weighted_giou_loss(baseline_config: dict) -> None:
    roi_heads: dict = baseline_config["roi_heads"]
    frames = [torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(2))]
    targets = [
        {
            "boxes": torch.tensor([[4.0, 6.0, 20.0, 18.0], [30.0, 10.0, 44.0, 30.0]]),
            "labels": torch.tensor([3, 1]),
        }
    ]
    plain = build_faster_rcnn(baseline_config | TINY_INPUT)
    losses = {}
    for name, roi_heads_config in (
        ("smooth_l1", roi_heads),
        ("giou", roi_heads | {"box_loss": "giou", "giou_loss_weight": 1.0}),
        ("giou_x2", roi_heads | {"box_loss": "giou", "giou_loss_weight": 2.0}),
    ):
        model = copy.deepcopy(plain)
        set_box_regression_loss(model, roi_heads_config)
        torch.manual_seed(0)
        losses[name] = model.train()(frames, targets)
    assert set(losses["giou"]) == set(losses["smooth_l1"])
    for key in ("loss_classifier", "loss_objectness", "loss_rpn_box_reg"):
        assert torch.equal(losses["giou"][key], losses["smooth_l1"][key])
    giou_loss = losses["giou"]["loss_box_reg"]
    assert giou_loss.item() > 0 and giou_loss.item() != losses["smooth_l1"]["loss_box_reg"].item()
    assert losses["giou_x2"]["loss_box_reg"].item() == pytest.approx(2 * giou_loss.item())


def test_box_loss_must_be_known_and_configured(baseline_config: dict) -> None:
    model = build_faster_rcnn(baseline_config | TINY_INPUT)
    with pytest.raises(ValueError, match="box_loss must be one of"):
        set_box_regression_loss(model, baseline_config["roi_heads"] | {"box_loss": "l1"})
    for field in ("box_loss", "giou_loss_weight"):
        roi_heads = {k: v for k, v in baseline_config["roi_heads"].items() if k != field}
        with pytest.raises(KeyError, match=field):
            set_box_regression_loss(model, roi_heads)
