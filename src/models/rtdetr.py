import re

import numpy as np
import polars as pl
import torch
from torchvision.ops import box_convert
from transformers import RTDetrV2ForObjectDetection

from src.data.classes import CLASS_NAMES

SUPPORTED_ARCHS = ("PekingU/rtdetr_v2_r50vd", "PekingU/rtdetr_v2_r18vd")
SUPPORTED_INIT = "coco"
PAD_MULTIPLE = 32
MAX_DETECTIONS_PER_IMG = 500
BACKBONE_UNITS = 5
LAYER_SUFFIX = re.compile(r"_(aux|dn)_\d+$")
FrameSize = tuple[int, int]


def compute_input_size(height: int, width: int, min_size: int, max_size: int) -> FrameSize:
    """Compute the resized frame size that keeps the aspect ratio.

    The shorter side is scaled to `min_size` unless the longer side would exceed `max_size`,
    as in torchvision's `GeneralizedRCNNTransform`
    (https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py).
    Sides are rounded to the nearest pixel, so the shorter side is exactly `min_size` whenever
    `max_size` does not bind.

    Args:
        height: Frame height in px.
        width: Frame width in px.
        min_size: Target length of the shorter side in px.
        max_size: Upper bound of the longer side in px.

    Returns:
        The resized (height, width) in px.
    """
    scale = min(min_size / min(height, width), max_size / max(height, width))
    return round(height * scale), round(width * scale)


def resize_and_pad_frames(
    frames: list[torch.Tensor], min_size: int, max_size: int
) -> tuple[torch.Tensor, list[FrameSize]]:
    """Resize frames with `compute_input_size` and pad them into one batch tensor.

    Frames are resized bilinearly with antialiasing, like the checkpoint's image processor
    (PIL bilinear), and padded with zeros at the bottom and right to the largest resized size
    of the batch, rounded up to a multiple of 32, the largest feature stride. Pixel values
    stay in [0, 1] without mean and std normalization, as in the checkpoint's
    `preprocessor_config.json` (`do_rescale` true, `do_normalize` false).

    Args:
        frames: float tensors of shape (3, height, width) with values in [0, 1].
        min_size: Target length of the shorter side in px.
        max_size: Upper bound of the longer side in px.

    Returns:
        Batch tensor of shape (n, 3, padded_height, padded_width) and the resized (height,
        width) of each frame.
    """
    resized_sizes = [compute_input_size(*frame.shape[-2:], min_size, max_size) for frame in frames]
    padded_height, padded_width = (
        -(-max(sizes) // PAD_MULTIPLE) * PAD_MULTIPLE for sizes in zip(*resized_sizes)
    )
    batch = frames[0].new_zeros((len(frames), 3, padded_height, padded_width))
    for i, (frame, (height, width)) in enumerate(zip(frames, resized_sizes, strict=True)):
        batch[i, :, :height, :width] = torch.nn.functional.interpolate(
            frame[None], size=(height, width), mode="bilinear", align_corners=False, antialias=True
        )[0]
    return batch, resized_sizes


def compute_canvas_scales(
    original_size: FrameSize, resized_size: FrameSize, padded_size: FrameSize
) -> torch.Tensor:
    """Compute per-coordinate factors from original pixels to normalized padded coordinates.

    RT-DETR places its anchors over the whole feature map, so its normalized boxes refer to
    the padded input, not to the resized frame inside it.

    Args:
        original_size: Frame (height, width) in px.
        resized_size: Resized frame (height, width) in px.
        padded_size: Padded batch (height, width) in px.

    Returns:
        Tensor of shape (4,) multiplying `x1, y1, x2, y2` in original px.
    """
    x_scale = resized_size[1] / original_size[1] / padded_size[1]
    y_scale = resized_size[0] / original_size[0] / padded_size[0]
    return torch.tensor([x_scale, y_scale, x_scale, y_scale])


def convert_targets_to_rtdetr(
    targets: list[dict[str, torch.Tensor]],
    original_sizes: list[FrameSize],
    resized_sizes: list[FrameSize],
    padded_size: FrameSize,
) -> list[dict[str, torch.Tensor]]:
    """Convert dataset targets to the RT-DETR label format.

    Args:
        targets: Per frame, `boxes` (n, 4) with `x1, y1, x2, y2` in original px and 0-based
            `labels` (n,); other keys are dropped.
        original_sizes: Frame (height, width) in px.
        resized_sizes: Resized frame (height, width) in px.
        padded_size: Padded batch (height, width) in px.

    Returns:
        Per frame, `class_labels` (n,) with the same 0-based labels and `boxes` (n, 4) with
        `cx, cy, w, h` normalized to the padded input.
    """
    converted = []
    for target, original_size, resized_size in zip(
        targets, original_sizes, resized_sizes, strict=True
    ):
        boxes = target["boxes"]
        scales = compute_canvas_scales(original_size, resized_size, padded_size)
        normalized = boxes * scales.to(boxes.device, boxes.dtype)
        converted.append(
            {"class_labels": target["labels"], "boxes": box_convert(normalized, "xyxy", "cxcywh")}
        )
    return converted


def convert_rtdetr_outputs(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    original_sizes: list[FrameSize],
    resized_sizes: list[FrameSize],
    padded_size: FrameSize,
    score_thresh: float,
    detections_per_img: int,
) -> list[dict[str, torch.Tensor]]:
    """Select the top-scoring (query, class) pairs of each frame and box them in original px.

    As in the official RT-DETR post-processor
    (https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py),
    sigmoid scores of all queries and classes are ranked together and the top
    `detections_per_img` are kept without NMS, so one query may yield several classes. Pairs
    with a score of at most `score_thresh` are dropped and boxes are clipped to the frame.

    Args:
        logits: Tensor (n, queries, classes) of class logits.
        pred_boxes: Tensor (n, queries, 4) with `cx, cy, w, h` normalized to the padded input.
        original_sizes: Frame (height, width) in px.
        resized_sizes: Resized frame (height, width) in px.
        padded_size: Padded batch (height, width) in px.
        score_thresh: Scores must exceed this to be kept.
        detections_per_img: Number of ranked pairs per frame before the score threshold.

    Returns:
        Per frame, `boxes` (k, 4) with `x1, y1, x2, y2` in original px, `scores` (k,), and
        0-based `labels` (k,), in descending score order.
    """
    num_classes = logits.shape[-1]
    scores, indices = logits.float().sigmoid().flatten(1).topk(detections_per_img, dim=1)
    boxes = box_convert(pred_boxes.float(), "cxcywh", "xyxy")
    detections = []
    for i, (original_size, resized_size) in enumerate(
        zip(original_sizes, resized_sizes, strict=True)
    ):
        scales = compute_canvas_scales(original_size, resized_size, padded_size)
        frame_boxes = boxes[i, indices[i] // num_classes] / scales.to(boxes.device)
        height, width = original_size
        frame_boxes[:, 0::2] = frame_boxes[:, 0::2].clamp(0, width)
        frame_boxes[:, 1::2] = frame_boxes[:, 1::2].clamp(0, height)
        is_kept = scores[i] > score_thresh
        detections.append(
            {
                "boxes": frame_boxes[is_kept],
                "scores": scores[i][is_kept],
                "labels": (indices[i] % num_classes)[is_kept],
            }
        )
    return detections


def group_rtdetr_losses(loss_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sum the weighted RT-DETR losses of all decoder layers, the encoder, and denoising.

    Args:
        loss_dict: Weighted losses from the Hugging Face model, e.g. `loss_vfl`,
            `loss_vfl_aux_0`, and `loss_vfl_dn_0`.

    Returns:
        One total per loss term (`loss_vfl`, `loss_bbox`, `loss_giou`); their sum equals the
        model's training loss.
    """
    grouped: dict[str, torch.Tensor] = {}
    for name, value in loss_dict.items():
        base = LAYER_SUFFIX.sub("", name)
        grouped[base] = grouped[base] + value if base in grouped else value
    return grouped


class RTDetrDetector(torch.nn.Module):
    """RT-DETRv2 with the input transform and output format of a torchvision detector.

    Calling the module runs `compute_losses_or_detections` on a list of float frames
    (3, height, width) in [0, 1]. In training mode it also takes targets with `boxes`
    (`x1, y1, x2, y2` in px) and 0-based `labels` and returns the grouped losses of
    `group_rtdetr_losses`; in eval mode it returns per frame `boxes` in px, `scores`, and
    0-based `labels` from `convert_rtdetr_outputs`.

    Args:
        detector: Hugging Face RT-DETRv2 detector with our classes.
        min_size: Target length of the shorter frame side in px.
        max_size: Upper bound of the longer frame side in px.
        score_thresh: Scores must exceed this to be kept at inference.
        detections_per_img: Ranked (query, class) pairs kept per frame at inference.

    Attributes:
        detector: The Hugging Face detector.
        min_size: Target length of the shorter frame side in px.
        max_size: Upper bound of the longer frame side in px.
        score_thresh: Minimum score at inference.
        detections_per_img: Maximum detections per frame.
    """

    def __init__(
        self,
        detector: RTDetrV2ForObjectDetection,
        min_size: int,
        max_size: int,
        score_thresh: float,
        detections_per_img: int,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.min_size = min_size
        self.max_size = max_size
        self.score_thresh = score_thresh
        self.detections_per_img = detections_per_img

    def compute_losses_or_detections(
        self, frames: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Compute the training losses or, in eval mode, the detections of a batch of frames.

        Args:
            frames: float tensors of shape (3, height, width) with values in [0, 1].
            targets: Per frame, `boxes` (n, 4) with `x1, y1, x2, y2` in px and 0-based
                `labels` (n,); required in training mode, ignored in eval mode.

        Returns:
            In training mode, the grouped losses of `group_rtdetr_losses`; in eval mode, per
            frame `boxes`, `scores`, and `labels` from `convert_rtdetr_outputs`.
        """
        original_sizes = [(frame.shape[-2], frame.shape[-1]) for frame in frames]
        pixel_values, resized_sizes = resize_and_pad_frames(frames, self.min_size, self.max_size)
        padded_size = (pixel_values.shape[-2], pixel_values.shape[-1])
        if self.training:
            if targets is None:
                raise ValueError("RT-DETR needs targets in training mode")
            labels = convert_targets_to_rtdetr(targets, original_sizes, resized_sizes, padded_size)
            outputs = self.detector(pixel_values=pixel_values, labels=labels)
            return group_rtdetr_losses(outputs.loss_dict)
        outputs = self.detector(pixel_values=pixel_values)
        return convert_rtdetr_outputs(
            outputs.logits,
            outputs.pred_boxes,
            original_sizes,
            resized_sizes,
            padded_size,
            self.score_thresh,
            self.detections_per_img,
        )

    forward = compute_losses_or_detections


def freeze_backbone_stages(detector: RTDetrV2ForObjectDetection, trainable_layers: int) -> None:
    """Freeze the stem and the first ResNet stages of the backbone.

    Counts as torchvision's `trainable_backbone_layers`: of the stem and the 4 stages, the
    last `trainable_layers` stay trainable. The official RT-DETRv2 R50 recipe (`freeze_at: 0`)
    freezes only the stem, which is 4 here
    (https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/nn/backbone/presnet.py).

    Args:
        detector: Hugging Face RT-DETRv2 detector.
        trainable_layers: Trainable backbone units counted from the last, 0 to 5.
    """
    if not 0 <= trainable_layers <= BACKBONE_UNITS:
        raise ValueError(
            f"trainable_backbone_layers must be 0 to {BACKBONE_UNITS}, got {trainable_layers}"
        )
    backbone = detector.model.backbone.model
    units = [backbone.embedder, *backbone.encoder.stages]
    for unit in units[: BACKBONE_UNITS - trainable_layers]:
        unit.requires_grad_(False)


def build_rtdetr(config: dict) -> RTDetrDetector:
    """Build RT-DETRv2 from its COCO checkpoint with new classification heads for our classes.

    All COCO weights are loaded except the shapes that depend on the number of classes (the
    decoder and encoder class heads and the denoising label embedding), which Hugging Face
    initializes anew (https://huggingface.co/docs/transformers/model_doc/rt_detr_v2), seeded
    with the config seed. Labels are our 0-based class ids without a background class. Query,
    denoising, matcher, and loss settings come from the config.

    Args:
        config: Resolved configuration with `arch`, `revision`, `init`,
            `trainable_backbone_layers`, `min_size`, `max_size`, `seed`, and the `[queries]`,
            `[matcher]`, `[loss]`, and `[inference]` tables of `config/rtdetr/baseline.toml`.

    Returns:
        The wrapped detector with 8 output classes.
    """
    if config["arch"] not in SUPPORTED_ARCHS or config["init"] != SUPPORTED_INIT:
        raise ValueError(
            f"RT-DETR supports arch {SUPPORTED_ARCHS} with init {SUPPORTED_INIT!r} only, "
            f"got {config['arch']!r} with {config['init']!r}"
        )
    queries, matcher, loss = config["queries"], config["matcher"], config["loss"]
    inference = config["inference"]
    max_detections = min(queries["num_queries"], MAX_DETECTIONS_PER_IMG)
    if not 1 <= inference["detections_per_img"] <= max_detections:
        raise ValueError(
            f"detections_per_img must be 1 to {max_detections} (num_queries and the "
            f"{MAX_DETECTIONS_PER_IMG} box limit), got {inference['detections_per_img']}"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        detector = RTDetrV2ForObjectDetection.from_pretrained(
            config["arch"],
            revision=config["revision"],
            id2label=dict(enumerate(CLASS_NAMES)),
            label2id={name: i for i, name in enumerate(CLASS_NAMES)},
            ignore_mismatched_sizes=True,
            num_queries=queries["num_queries"],
            num_denoising=queries["num_denoising"],
            label_noise_ratio=queries["label_noise_ratio"],
            box_noise_scale=queries["box_noise_scale"],
            matcher_class_cost=matcher["class_cost"],
            matcher_bbox_cost=matcher["bbox_cost"],
            matcher_giou_cost=matcher["giou_cost"],
            matcher_alpha=matcher["focal_alpha"],
            matcher_gamma=matcher["focal_gamma"],
            weight_loss_vfl=loss["vfl_weight"],
            weight_loss_bbox=loss["bbox_weight"],
            weight_loss_giou=loss["giou_weight"],
            focal_loss_alpha=loss["focal_alpha"],
            focal_loss_gamma=loss["focal_gamma"],
            auxiliary_loss=loss["auxiliary_loss"],
        )
    freeze_backbone_stages(detector, config["trainable_backbone_layers"])
    return RTDetrDetector(
        detector,
        config["min_size"],
        config["max_size"],
        inference["score_thresh"],
        inference["detections_per_img"],
    )


def predict_query_boxes(model: RTDetrDetector, frame: torch.Tensor) -> pl.DataFrame:
    """Predict the box and the best class of every object query for one frame.

    Inference ranks (query, class) pairs; this keeps one row per query instead, with its
    highest-scoring class, so that the queries can be drawn as the decoder outputs them.

    Args:
        model: Detector from `build_rtdetr` in eval mode.
        frame: float tensor of shape (3, height, width) with values in [0, 1], on the
            model's device.

    Returns:
        One row per query in descending score order with `rank` (0 for the highest score),
        0-based `class_id`, sigmoid `score`, and pixel `x`, `y`, `w`, `h` (top-left corner)
        in the original frame, clipped to it.
    """
    height, width = frame.shape[-2], frame.shape[-1]
    with torch.inference_mode():
        pixel_values, [resized_size] = resize_and_pad_frames(
            [frame], model.min_size, model.max_size
        )
        outputs = model.detector(pixel_values=pixel_values)
    scores, labels = outputs.logits[0].float().sigmoid().max(dim=-1)
    padded_size = (pixel_values.shape[-2], pixel_values.shape[-1])
    scales = compute_canvas_scales((height, width), resized_size, padded_size)
    boxes = box_convert(outputs.pred_boxes[0].float(), "cxcywh", "xyxy") / scales.to(frame.device)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    order = scores.argsort(descending=True)
    xyxy = boxes[order].cpu().double().numpy()
    return pl.DataFrame(
        {
            "rank": np.arange(len(order)),
            "class_id": labels[order].cpu().numpy().astype(np.int64),
            "score": scores[order].cpu().double().numpy(),
            "x": xyxy[:, 0],
            "y": xyxy[:, 1],
            "w": xyxy[:, 2] - xyxy[:, 0],
            "h": xyxy[:, 3] - xyxy[:, 1],
        }
    )
