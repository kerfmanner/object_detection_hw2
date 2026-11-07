import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.transform import resize_boxes
from tqdm.auto import tqdm

from src.data.annotations import BOX_COLUMNS
from src.inference.detections import build_prediction_table
from src.models.torchvision_labels import (
    TORCHVISION_FAMILIES,
    shift_detections_from_torchvision,
)

MODEL_STATE_KEY = "model"


def select_device() -> torch.device:
    """Pick the first GPU when CUDA is available, otherwise the CPU.

    Returns:
        The device for training and inference.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_frames_to_inputs(
    frames: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Move uint8 RGB frames to the device and convert them to float there.

    Args:
        frames: uint8 tensors of shape (3, height, width).
        device: Device of the model.

    Returns:
        float32 tensors of the same shapes on `device` with values in [0, 1].
    """
    return [frame.to(device, non_blocking=True).float().div_(255) for frame in frames]


def convert_torchvision_detections(detections: dict[str, torch.Tensor]) -> np.ndarray:
    """Convert the detections of one frame to classes, scores, and top-left boxes.

    Boxes that torchvision clipped to zero width or height at the frame border are dropped;
    Faster R-CNN already removes them, RetinaNet does not.

    Args:
        detections: `boxes` (k, 4) with `x1, y1, x2, y2` in px of the original frame,
            `scores` (k,), and 0-based `labels` (k,).

    Returns:
        Array of shape (n, 6), n <= k, with the 0-based class id, score, and `x, y, w, h`
        in px.
    """
    xyxy = detections["boxes"].double().cpu().numpy()
    converted = np.column_stack(
        [
            detections["labels"].cpu().numpy(),
            detections["scores"].double().cpu().numpy(),
            xyxy[:, :2],
            xyxy[:, 2:] - xyxy[:, :2],
        ]
    )
    return converted[(converted[:, 4] > 0) & (converted[:, 5] > 0)]


def detect_frames(
    model: torch.nn.Module,
    images: pl.DataFrame,
    device: torch.device,
    warmup_images: int,
    has_background_label: bool,
) -> tuple[pl.DataFrame, float]:
    """Detect objects on every frame of an image table with a torchvision-style detector.

    Frames go through the model one at a time. The model is put in eval mode on `device`.
    The time per image covers moving the frame to the device, the uint8 to float conversion,
    and the forward pass with resizing and post-processing (NMS for torchvision), with the
    GPU synchronized before and after; decoding the file is not timed, as for YOLO.
    `warmup_images` untimed passes on the first frame come first.

    Args:
        model: Detector taking a list of float frames and returning `boxes`, `scores`, and
            `labels` per frame, already on `device`.
        images: Image table with `image_id` and absolute `file_path`.
        device: Device of the model.
        warmup_images: Number of untimed passes before measuring.
        has_background_label: Whether the model predicts torchvision labels with background
            0, which are shifted back to 0-based class ids; otherwise labels are 0-based.

    Returns:
        Predictions with `image_id`, 0-based `class_id`, `score`, and pixel `x`, `y`, `w`,
        `h` (top-left corner), and the mean inference time in ms per image.
    """
    model.eval()
    is_cuda = device.type == "cuda"
    file_paths = images["file_path"].to_list()
    frame_detections, seconds = [], []
    with torch.inference_mode():
        warmup_frame = decode_image(file_paths[0], mode=ImageReadMode.RGB)
        for _ in range(warmup_images):
            model(convert_frames_to_inputs([warmup_frame], device))
        for file_path in tqdm(file_paths, desc="Predicting frames", unit="frame"):
            frame = decode_image(file_path, mode=ImageReadMode.RGB)
            if is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(convert_frames_to_inputs([frame], device))
            if is_cuda:
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            if has_background_label:
                outputs = shift_detections_from_torchvision(outputs)
            [detections] = outputs
            frame_detections.append(convert_torchvision_detections(detections))
    predictions = build_prediction_table(images["image_id"], frame_detections)
    return predictions, 1000 * float(np.mean(seconds))


def predict_torchvision_frames(
    model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
) -> tuple[pl.DataFrame, float]:
    """Predict every frame of an image table with a checkpoint of the shared training loop.

    Used for the torchvision families and RT-DETR. The score threshold, NMS IoU, and
    detections per frame of `[inference]` were set in the model by its builder.

    Args:
        model: Detector from `build_model` with the run's configuration.
        checkpoint_path: Path of a checkpoint written by the shared training loop.
        images: Image table with `image_id` and absolute `file_path`.
        config: Resolved run configuration with `run.family` and `[inference]`
            `warmup_images`.

    Returns:
        Predictions with `image_id`, 0-based `class_id`, `score`, and pixel `x`, `y`, `w`,
        `h` (top-left corner), and the mean inference time in ms per image.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint[MODEL_STATE_KEY])
    device = select_device()
    model.to(device)
    return detect_frames(
        model,
        images,
        device,
        config["inference"]["warmup_images"],
        config["run"]["family"] in TORCHVISION_FAMILIES,
    )


def predict_rpn_proposals(
    detector: FasterRCNN, frame: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Predict the region proposals of one frame with a Faster R-CNN in eval mode.

    The frame goes through the detector's resize transform, backbone, and RPN, the first
    half of torchvision's `GeneralizedRCNN.forward`, and the proposals are scaled back to the
    original frame like the final detections:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py

    Args:
        detector: Faster R-CNN on `device` in eval mode, so that the RPN keeps its
            `post_nms_top_n_test` proposals.
        frame: uint8 tensor of shape (3, height, width).
        device: Device of the detector.

    Returns:
        Array of shape (k, 4) with `x, y, w, h` in px of the original frame, in descending
        objectness order, the order of the RPN's NMS output.
    """
    with torch.inference_mode():
        images, _ = detector.transform(convert_frames_to_inputs([frame], device))
        features = detector.backbone(images.tensors)
        [proposals], _ = detector.rpn(images, features)
        boxes = resize_boxes(proposals, list(images.image_sizes[0]), list(frame.shape[-2:]))
    xyxy = boxes.double().cpu().numpy()
    return np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])


def tabulate_rpn_proposals(
    detector: FasterRCNN, images: pl.DataFrame, device: torch.device
) -> pl.DataFrame:
    """Predict the region proposals of every frame of an image table.

    Args:
        detector: Faster R-CNN on `device` in eval mode.
        images: Image table with `image_id` and absolute `file_path`.
        device: Device of the detector.

    Returns:
        One row per proposal with `image_id`, `rank` (0 for the highest objectness in its
        frame), and pixel `x`, `y`, `w`, `h` (top-left corner) in the original frame.
    """
    frames = []
    rows = images.select("image_id", "file_path").iter_rows()
    for image_id, file_path in tqdm(rows, total=images.height, desc="Proposals", unit="frame"):
        frame = decode_image(file_path, mode=ImageReadMode.RGB)
        boxes = predict_rpn_proposals(detector, frame, device)
        frames.append(
            pl.DataFrame(
                {
                    "image_id": pl.Series([image_id] * len(boxes), dtype=images["image_id"].dtype),
                    "rank": np.arange(len(boxes)),
                    **{column: boxes[:, i] for i, column in enumerate(BOX_COLUMNS)},
                }
            )
        )
    return pl.concat(frames)
