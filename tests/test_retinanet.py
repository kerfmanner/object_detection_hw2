import copy
import math
from pathlib import Path

import polars as pl
import pytest
import torch
from torchvision.io import write_jpeg
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetClassificationHead
from torchvision.ops import sigmoid_focal_loss

from src.inference.torchvision_detectors import tabulate_anchor_matches
from src.models.retinanet import FocalLossClassificationHead, build_retinanet, set_focal_loss
from src.training.experiment import load_experiment_config
from tests.test_smoke import RETINANET_BASELINE

CLASSIFIER_KEY = "head.classification_head.cls_logits"
TINY_INPUT = {"min_size": 64, "max_size": 96}


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


def build_loss_inputs() -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor], list]:
    generator = torch.Generator().manual_seed(0)
    targets = [{"labels": torch.tensor([3, 8])}, {"labels": torch.tensor([1])}]
    head_outputs = {"cls_logits": torch.randn(2, 6, 9, generator=generator)}
    matched_idxs = [torch.tensor([0, -1, -2, 1, -1, -1]), torch.tensor([-1, -1, -1, -2, -1, -1])]
    return targets, head_outputs, matched_idxs


def test_focal_loss_head_reproduces_torchvision_at_its_defaults(
    baseline_config: dict, baseline_model: RetinaNet
) -> None:
    head = baseline_model.head.classification_head
    assert isinstance(head, RetinaNetClassificationHead)
    rng_state = torch.random.get_rng_state()
    focal = FocalLossClassificationHead(head, alpha=0.25, gamma=2.0)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    head_state, focal_state = head.state_dict(), focal.state_dict()
    assert head_state.keys() == focal_state.keys()
    assert all(torch.equal(head_state[key], focal_state[key]) for key in head_state)
    features = [torch.randn(1, 256, 4, 6, generator=torch.Generator().manual_seed(1))]
    with torch.inference_mode():
        assert torch.equal(focal(features), head(features))
    inputs = build_loss_inputs()
    assert torch.allclose(focal.compute_loss(*inputs), head.compute_loss(*inputs))


def test_focal_loss_head_uses_the_configured_alpha_and_gamma(
    baseline_model: RetinaNet,
) -> None:
    head = baseline_model.head.classification_head
    assert isinstance(head, RetinaNetClassificationHead)
    targets, head_outputs, matched_idxs = build_loss_inputs()
    loss = FocalLossClassificationHead(head, alpha=0.5, gamma=1.0).compute_loss(
        targets, head_outputs, matched_idxs
    )
    first_logits, second_logits = head_outputs["cls_logits"]
    first_target = torch.zeros(6, 9)
    first_target[0, 3] = first_target[3, 8] = 1.0
    first_valid = matched_idxs[0] != -2
    second_valid = matched_idxs[1] != -2
    first = sigmoid_focal_loss(
        first_logits[first_valid], first_target[first_valid], 0.5, 1.0, reduction="sum"
    )
    second = sigmoid_focal_loss(
        second_logits[second_valid], torch.zeros(5, 9), 0.5, 1.0, reduction="sum"
    )
    assert loss.item() == pytest.approx(((first / 2 + second / 1) / 2).item(), rel=1e-6)


def test_set_focal_loss_replaces_the_head_and_validates_the_table(
    baseline_config: dict, baseline_model: RetinaNet
) -> None:
    model = copy.deepcopy(baseline_model)
    focal_loss: dict = baseline_config["focal_loss"]
    set_focal_loss(model, focal_loss | {"gamma": 1.0})
    head = model.head.classification_head
    assert isinstance(head, FocalLossClassificationHead)
    assert (head.alpha, head.gamma) == (0.25, 1.0)
    model.load_state_dict(baseline_model.state_dict())
    for change in ({"alpha": 1.5}, {"gamma": -1.0}):
        with pytest.raises(ValueError, match="Focal loss needs"):
            set_focal_loss(model, focal_loss | change)
    for field in ("alpha", "gamma"):
        with pytest.raises(KeyError, match=field):
            set_focal_loss(model, {k: v for k, v in focal_loss.items() if k != field})


def test_focal_loss_trains_the_detector(baseline_config: dict) -> None:
    model = build_retinanet(baseline_config | TINY_INPUT)
    set_focal_loss(model, baseline_config["focal_loss"] | {"gamma": 1.0})
    frames = [torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(2))]
    targets = [{"boxes": torch.tensor([[4.0, 6.0, 20.0, 18.0]]), "labels": torch.tensor([3])}]
    losses = model.train()(frames, targets)
    assert set(losses) == {"classification", "bbox_regression"}
    losses["classification"].backward()
    assert model.head.classification_head.cls_logits.weight.grad is not None


def test_anchor_matches_follow_the_retinanet_matcher(baseline_config: dict, tmp_path: Path) -> None:
    detector = build_retinanet(baseline_config | TINY_INPUT).eval()
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for path in paths:
        write_jpeg(torch.zeros((3, 48, 64), dtype=torch.uint8), str(path))
    images = pl.DataFrame({"image_id": [1, 2], "file_path": [str(path) for path in paths]})
    objects = pl.DataFrame(
        {
            "image_id": [1, 1],
            "x": [4.0, 30.0],
            "y": [4.0, 10.0],
            "w": [20.0, 30.0],
            "h": [12.0, 30.0],
        }
    )
    frames, matched = tabulate_anchor_matches(detector, images, objects, torch.device("cpu"))
    assert frames["image_id"].to_list() == [1, 2] and frames["objects"].to_list() == [2, 0]
    totals = frames.select(pl.col("positives") + pl.col("negatives") + pl.col("ignored"))
    assert totals.to_series().to_list() == frames["anchors"].to_list()
    assert frames.row(1, named=True)["positives"] == 0
    assert matched.height == 2 and (matched["positive_anchors"] >= 1).all()
    assert matched["positive_anchors"].sum() == frames["positives"][0]
