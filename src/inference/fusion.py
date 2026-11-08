import numpy as np
import polars as pl
import torch
from torchvision.ops import batched_nms

from src.data.annotations import BOX_COLUMNS
from src.evaluation.coco_metrics import MAX_DETECTIONS_PER_IMAGE, PREDICTION_COLUMNS

FUSION_METHODS = ("wbf", "nms")
FRAME_KEY = ("image_id", "class_id")


def compute_box_ious(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute the IoU of one corner box with each of several corner boxes.

    Args:
        box: Array (4,) with `x1, y1, x2, y2` in px.
        boxes: Array (k, 4) in the same format.

    Returns:
        Array (k,) with IoU values in [0, 1].
    """
    width = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    height = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    intersection = width * height
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / ((box[2] - box[0]) * (box[3] - box[1]) + areas - intersection)


def cluster_weighted_boxes(
    xyxy: np.ndarray, scores: np.ndarray, weights: np.ndarray, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster the boxes of one class and frame as Weighted Boxes Fusion does.

    Follows section 3 of https://arxiv.org/abs/1910.13302: boxes are visited in descending
    score order; a box joins the cluster whose current fused box overlaps it most if that IoU
    exceeds `iou_thresh`, otherwise it starts a new cluster. A fused box is the average of its
    cluster's corners weighted by score times model weight, recomputed after every addition.

    Args:
        xyxy: Array (n, 4) with `x1, y1, x2, y2` in px, sorted by descending `scores`.
        scores: Array (n,) of detection scores.
        weights: Array (n,) with the weight of the model that produced each box.
        iou_thresh: A box joins a cluster only above this IoU with its fused box.

    Returns:
        Fused corner boxes (k, 4), the sum of score times weight per cluster (k,), and the
        sum of the box weights per cluster (k,).
    """
    weighted_scores = scores * weights
    fused = np.empty_like(xyxy)
    corner_sums = np.empty_like(xyxy)
    score_sums = np.empty(len(xyxy))
    weight_sums = np.empty(len(xyxy))
    count = 0
    for box, weighted_score, weight in zip(xyxy, weighted_scores, weights, strict=True):
        if count:
            ious = compute_box_ious(box, fused[:count])
            best = int(ious.argmax())
            if ious[best] > iou_thresh:
                corner_sums[best] += weighted_score * box
                score_sums[best] += weighted_score
                weight_sums[best] += weight
                fused[best] = corner_sums[best] / score_sums[best]
                continue
        corner_sums[count] = weighted_score * box
        score_sums[count] = weighted_score
        weight_sums[count] = weight
        fused[count] = box
        count += 1
    return fused[:count], score_sums[:count], weight_sums[:count]


def stack_member_predictions(members: list[pl.DataFrame], weights: list[float]) -> pl.DataFrame:
    """Concatenate the predictions of several models with each model's weight.

    Args:
        members: Prediction tables with `PREDICTION_COLUMNS`, one per model.
        weights: Positive weight of each model.

    Returns:
        All rows with `PREDICTION_COLUMNS` and float `weight`.
    """
    if len(members) != len(weights) or not members:
        raise ValueError(f"Need one weight per member, got {len(members)} and {len(weights)}")
    if min(weights) <= 0:
        raise ValueError(f"Member weights must be positive, got {weights}")
    return pl.concat(
        [
            table.select(PREDICTION_COLUMNS).with_columns(weight=pl.lit(float(weight)))
            for table, weight in zip(members, weights, strict=True)
        ]
    )


def keep_top_detections(predictions: pl.DataFrame, max_detections: int) -> pl.DataFrame:
    """Keep the highest-scoring detections of every frame.

    Args:
        predictions: Table with `image_id` and `score`.
        max_detections: Detections kept per frame.

    Returns:
        At most `max_detections` rows per frame, sorted by `image_id` and descending score.
    """
    return (
        predictions.sort("image_id", "score", descending=[False, True], maintain_order=True)
        .group_by("image_id", maintain_order=True)
        .head(max_detections)
    )


def fuse_weighted_boxes(
    members: list[pl.DataFrame],
    weights: list[float],
    iou_thresh: float,
    max_detections: int = MAX_DETECTIONS_PER_IMAGE,
) -> pl.DataFrame:
    """Fuse the predictions of several models or passes with Weighted Boxes Fusion.

    Clusters every class of every frame separately (`cluster_weighted_boxes`). The fused
    score is the sum of score times model weight over the cluster divided by the larger of
    the total model weight and the cluster's weight sum. With unit weights this is the
    paper's average score rescaled by min(T, |cluster|) / T for T models
    (https://arxiv.org/abs/1910.13302, section 3), so a box found by fewer models scores
    lower; model weights follow the reference implementation
    (https://github.com/ZFTurbo/Weighted-Boxes-Fusion). The generalization keeps scores in
    [0, 1] when one model puts several boxes into a cluster.

    Args:
        members: Prediction tables with `PREDICTION_COLUMNS`, one per model or pass, on the
            same frames.
        weights: Positive weight of each member.
        iou_thresh: IoU above which a box joins a cluster.
        max_detections: Fused detections kept per frame, highest scores first.

    Returns:
        Fused predictions with `PREDICTION_COLUMNS`, sorted by `image_id` and score.
    """
    stacked = stack_member_predictions(members, weights)
    total_weight = float(sum(weights))
    image_dtype = stacked.schema["image_id"]
    parts = []
    groups = stacked.sort([*FRAME_KEY, "score"], descending=[False, False, True]).partition_by(
        FRAME_KEY, as_dict=True, maintain_order=True
    )
    for (image_id, class_id), group in groups.items():
        boxes = group.select(BOX_COLUMNS).to_numpy().astype(np.float64)
        xyxy = np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1)
        fused, score_sums, weight_sums = cluster_weighted_boxes(
            xyxy, group["score"].to_numpy(), group["weight"].to_numpy(), iou_thresh
        )
        parts.append(
            pl.DataFrame(
                {
                    "image_id": pl.Series([image_id] * len(fused), dtype=image_dtype),
                    "class_id": np.full(len(fused), class_id, dtype=np.int64),
                    "score": score_sums / np.maximum(total_weight, weight_sums),
                    "x": fused[:, 0],
                    "y": fused[:, 1],
                    "w": fused[:, 2] - fused[:, 0],
                    "h": fused[:, 3] - fused[:, 1],
                }
            )
        )
    if not parts:
        return stacked.select(PREDICTION_COLUMNS)
    return keep_top_detections(pl.concat(parts), max_detections)


def fuse_with_nms(
    members: list[pl.DataFrame],
    weights: list[float],
    iou_thresh: float,
    max_detections: int = MAX_DETECTIONS_PER_IMAGE,
) -> pl.DataFrame:
    """Fuse the predictions of several models by concatenation and per-class NMS.

    The comparison method for Weighted Boxes Fusion: scores are multiplied by the model
    weight divided by the largest weight, and NMS keeps the highest-scoring box of every
    overlapping same-class group in a frame.

    Args:
        members: Prediction tables with `PREDICTION_COLUMNS`, one per model or pass.
        weights: Positive weight of each member.
        iou_thresh: Boxes overlapping a kept box of their class and frame above this IoU
            are removed.
        max_detections: Detections kept per frame, highest scores first.

    Returns:
        Kept predictions with `PREDICTION_COLUMNS`, sorted by `image_id` and score.
    """
    stacked = stack_member_predictions(members, weights).with_columns(
        pl.col("score") * pl.col("weight") / max(weights)
    )
    boxes = stacked.select(BOX_COLUMNS).to_numpy().astype(np.float64)
    xyxy = torch.from_numpy(np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1))
    groups = stacked.select(pl.struct(FRAME_KEY).rank("dense").cast(pl.Int64)).to_series()
    kept = batched_nms(
        xyxy,
        torch.tensor(stacked["score"].to_numpy()),
        torch.tensor(groups.to_numpy()),
        iou_thresh,
    )
    return keep_top_detections(stacked[kept.numpy()].select(PREDICTION_COLUMNS), max_detections)


def fuse_predictions(
    members: list[pl.DataFrame], weights: list[float], method: str, iou_thresh: float
) -> pl.DataFrame:
    """Fuse member predictions with the configured method, at most 500 per frame.

    Args:
        members: Prediction tables with `PREDICTION_COLUMNS`, one per model or pass.
        weights: Positive weight of each member.
        method: `wbf` (`fuse_weighted_boxes`) or `nms` (`fuse_with_nms`).
        iou_thresh: IoU threshold of the method.

    Returns:
        Fused predictions with `PREDICTION_COLUMNS`.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"Fusion method must be one of {FUSION_METHODS}, got {method!r}")
    fuse = fuse_weighted_boxes if method == "wbf" else fuse_with_nms
    return fuse(members, weights, iou_thresh)
