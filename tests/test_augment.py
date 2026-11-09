from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.annotations import AnnotationTables
from src.data.augment import (
    augment_training_frame,
    build_train_transform,
    downscale_frame,
    draw_input_sizes,
    flip_frame,
    mask_ignored_regions,
)
from src.data.dataset import VisDroneDataset
from src.models.registry import build_model
from src.training.experiment import load_experiment_config
from src.training.torchvision_loop import check_training_budget, set_input_size
from tests.test_smoke import FASTER_RCNN_BASELINE, RTDETR_BASELINE

FILL = [118, 124, 120]


@pytest.fixture(scope="module")
def rtdetr_config() -> dict:
    return load_experiment_config(RTDETR_BASELINE)


def test_ignored_regions_are_filled_except_the_pixels_of_labelled_boxes() -> None:
    image = torch.zeros((3, 20, 30), dtype=torch.uint8)
    ignored = torch.tensor([[2.0, 2.0, 12.0, 10.0]])
    objects = torch.tensor([[8.2, 5.0, 15.5, 8.0]])
    masked = mask_ignored_regions(image, ignored, objects, FILL)
    is_filled = (masked == torch.tensor(FILL, dtype=torch.uint8)[:, None, None]).all(dim=0)
    expected = torch.zeros((20, 30), dtype=torch.bool)
    expected[2:10, 2:12] = True
    expected[5:8, 8:16] = False
    assert torch.equal(is_filled, expected)
    assert torch.equal(image, torch.zeros_like(image))


def test_flipping_mirrors_the_frame_and_all_boxes_and_twice_is_the_identity() -> None:
    image = torch.arange(3 * 4 * 6, dtype=torch.uint8).reshape(3, 4, 6)
    target = {
        "boxes": torch.tensor([[1.0, 0.0, 3.0, 2.0]]),
        "labels": torch.tensor([2]),
        "ignored_boxes": torch.tensor([[0.0, 1.0, 6.0, 4.0]]),
    }
    flipped, flipped_target = flip_frame(image, target)
    assert torch.equal(flipped[..., 0], image[..., -1])
    assert flipped_target["boxes"].tolist() == [[3.0, 0.0, 5.0, 2.0]]
    assert flipped_target["ignored_boxes"].tolist() == [[0.0, 1.0, 6.0, 4.0]]
    assert flipped_target["labels"] is target["labels"]
    again, again_target = flip_frame(flipped, flipped_target)
    assert torch.equal(again, image) and torch.equal(again_target["boxes"], target["boxes"])


def test_the_baseline_trains_on_unmodified_frames(rtdetr_config: dict) -> None:
    assert build_train_transform(rtdetr_config, is_clean=False) is None
    assert build_train_transform(rtdetr_config, is_clean=True) is None
    assert rtdetr_config["ignored_regions"]["fill"] == FILL


def test_clean_epochs_keep_only_the_masking_of_ignored_regions(rtdetr_config: dict) -> None:
    config = rtdetr_config | {
        "augment": rtdetr_config["augment"] | {"hflip": True, "photometric": True},
        "ignored_regions": rtdetr_config["ignored_regions"] | {"mask": True},
    }
    augmented = build_train_transform(config, is_clean=False)
    clean = build_train_transform(config, is_clean=True)
    assert isinstance(augmented, partial) and isinstance(clean, partial)
    assert augmented.keywords["is_flip_enabled"] and augmented.keywords["photometric"]
    assert clean.keywords == {
        "mask_fill": FILL,
        "photometric": None,
        "low_res_prob": 0.0,
        "low_res_factor": 0.5,
        "is_flip_enabled": False,
    }
    flip_only = config | {"ignored_regions": config["ignored_regions"] | {"mask": False}}
    assert build_train_transform(flip_only, is_clean=True) is None
    with pytest.raises(ValueError, match="fill must be three values"):
        build_train_transform(
            config | {"ignored_regions": {"mask": True, "fill": [0, 300, 0]}}, is_clean=True
        )


def test_flip_is_drawn_per_frame_and_keeps_boxes_on_the_object() -> None:
    image = torch.zeros((3, 10, 20), dtype=torch.uint8)
    image[:, 2:4, 1:5] = 255
    target = {
        "boxes": torch.tensor([[1.0, 2.0, 5.0, 4.0]]),
        "labels": torch.tensor([0]),
        "ignored_boxes": torch.zeros((0, 4)),
    }
    torch.manual_seed(0)
    lefts = []
    for _ in range(40):
        augmented, augmented_target = augment_training_frame(
            image, target, None, None, 0.0, 0.5, True
        )
        x1, y1, x2, y2 = augmented_target["boxes"][0].int().tolist()
        assert (augmented[:, y1:y2, x1:x2] == 255).all()
        lefts.append(x1)
    assert set(lefts) == {1, 15}


def test_downscaling_shrinks_the_frame_and_keeps_boxes_on_the_object() -> None:
    image = torch.zeros((3, 40, 80), dtype=torch.uint8)
    image[:, 8:24, 16:48] = 255
    target = {
        "boxes": torch.tensor([[16.0, 8.0, 48.0, 24.0]]),
        "labels": torch.tensor([4]),
        "ignored_boxes": torch.tensor([[0.0, 0.0, 80.0, 40.0]]),
    }
    shrunk, shrunk_target = downscale_frame(image, target, 0.5)
    assert shrunk.shape == (3, 20, 40) and shrunk.dtype == torch.uint8
    assert shrunk_target["boxes"].tolist() == [[8.0, 4.0, 24.0, 12.0]]
    assert shrunk_target["ignored_boxes"].tolist() == [[0.0, 0.0, 40.0, 20.0]]
    assert shrunk_target["labels"] is target["labels"]
    assert (shrunk[:, 5:11, 9:23] == 255).all() and shrunk[:, 14:, :].max() == 0


def test_low_resolution_is_drawn_with_its_probability_and_off_in_clean_epochs(
    rtdetr_config: dict,
) -> None:
    image = torch.zeros((3, 40, 80), dtype=torch.uint8)
    target = {
        "boxes": torch.tensor([[16.0, 8.0, 48.0, 24.0]]),
        "labels": torch.tensor([0]),
        "ignored_boxes": torch.zeros((0, 4)),
    }
    torch.manual_seed(0)
    heights = {
        augment_training_frame(image, target, None, None, 0.5, 0.5, False)[0].shape[1]
        for _ in range(40)
    }
    assert heights == {20, 40}
    always = augment_training_frame(image, target, None, None, 1.0, 0.25, False)[0]
    assert always.shape == (3, 10, 20)
    config = rtdetr_config | {"augment": rtdetr_config["augment"] | {"low_res_prob": 0.25}}
    augmented = build_train_transform(config, is_clean=False)
    assert isinstance(augmented, partial) and augmented.keywords["low_res_prob"] == 0.25
    assert build_train_transform(config, is_clean=True) is None
    for invalid in ({"low_res_prob": 1.5}, {"low_res_factor": 1.0}, {"low_res_factor": 0.0}):
        with pytest.raises(ValueError, match="low_res_prob must be in"):
            build_train_transform(
                rtdetr_config | {"augment": rtdetr_config["augment"] | invalid}, is_clean=False
            )


def test_multi_scale_sizes_come_from_the_list_and_scale_the_longer_side_bound() -> None:
    scales = [832, 1024, 1024, 1216]
    sizes = draw_input_sizes(scales, 1024, 1824, 200, np.random.default_rng([42, 1]))
    assert len(sizes) == 200 and {size for size, _ in sizes} == {832, 1024, 1216}
    assert dict(sizes) == {832: 1482, 1024: 1824, 1216: 2166}
    assert sizes == draw_input_sizes(scales, 1024, 1824, 200, np.random.default_rng([42, 1]))
    assert sizes != draw_input_sizes(scales, 1024, 1824, 200, np.random.default_rng([42, 2]))
    with pytest.raises(ValueError, match="train_scales must be positive"):
        draw_input_sizes([], 1024, 1824, 1, np.random.default_rng(0))


@pytest.mark.parametrize("baseline_path", [FASTER_RCNN_BASELINE, RTDETR_BASELINE])
def test_the_input_size_of_a_detector_can_be_set_per_batch(baseline_path: Path) -> None:
    config = load_experiment_config(baseline_path)
    model = build_model(config | {"min_size": 64, "max_size": 96})
    set_input_size(model, 128, 192)
    model.eval()
    with torch.inference_mode():
        model([torch.rand(3, 40, 80)])
    transform = getattr(model, "transform", None)
    if transform is None:
        assert (model.min_size, model.max_size) == (128, 192)
    else:
        assert (transform.min_size, transform.max_size) == ((128,), 192)
        images, _ = transform([torch.rand(3, 40, 80)])
        assert images.image_sizes == [(96, 192)]
    with pytest.raises(TypeError, match="Cannot set the input size"):
        set_input_size(torch.nn.Linear(1, 1), 128, 192)


def test_clean_epochs_must_leave_an_augmented_epoch(rtdetr_config: dict) -> None:
    check_training_budget(
        rtdetr_config | {"augment": rtdetr_config["augment"] | {"clean_epochs": 6}}
    )
    for clean_epochs in (7, -1, 1.0):
        augment = rtdetr_config["augment"] | {"clean_epochs": clean_epochs}
        with pytest.raises(ValueError, match="clean_epochs must be an integer from 0"):
            check_training_budget(rtdetr_config | {"augment": augment})


def blank_frame(
    image: torch.Tensor, target: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return torch.zeros_like(image), target | {"labels": target["labels"] + 1}


def test_the_dataset_applies_its_transform_to_frame_and_target(
    synthetic_annotations: AnnotationTables,
) -> None:
    plain = VisDroneDataset(synthetic_annotations, "train", frame_step=1)
    blank = VisDroneDataset(synthetic_annotations, "train", frame_step=1, transform=blank_frame)
    image, target = plain[0]
    blank_image, blank_target = blank[0]
    assert image.any() and not blank_image.any() and blank_image.shape == image.shape
    assert torch.equal(blank_target["labels"], target["labels"] + 1)
    assert torch.equal(blank_target["boxes"], target["boxes"])
