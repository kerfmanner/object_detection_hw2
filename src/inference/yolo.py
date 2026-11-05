import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
from ultralytics.utils.patches import imread

from src.data.annotations import BOX_COLUMNS
from src.data.classes import CLASS_NAMES


def convert_yolo_boxes(boxes: Boxes) -> np.ndarray:
    """Convert the Ultralytics boxes of one frame to classes, scores, and top-left boxes.

    Boxes that Ultralytics clipped to zero width or height at the frame border are dropped.

    Args:
        boxes: Boxes of one prediction result, in pixels of the original frame.

    Returns:
        Array of shape (n, 6) with the 0-based class id, score, and `x, y, w, h` in px.
    """
    boxes = boxes.cpu().numpy()
    xyxy = boxes.xyxy.astype(np.float64)
    detections = np.column_stack([boxes.cls, boxes.conf, xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
    return detections[(detections[:, 4] > 0) & (detections[:, 5] > 0)]


def load_yolo_checkpoint(checkpoint_path: Path) -> YOLO:
    """Load a trained YOLO checkpoint and check that its classes are ours in our order.

    Args:
        checkpoint_path: Path of a checkpoint written by Ultralytics training.

    Returns:
        The detector, whose class index equals our 0-based `class_id`.
    """
    detector = YOLO(checkpoint_path)
    expected = dict(enumerate(CLASS_NAMES))
    if detector.names != expected:
        raise ValueError(
            f"{checkpoint_path} predicts classes {detector.names}, expected {expected}"
        )
    return detector


def read_bgr_frame(file_path: str) -> np.ndarray:
    """Read a frame with the decoder Ultralytics uses for training images.

    Args:
        file_path: Path of the frame.

    Returns:
        Array of shape (height, width, 3) in BGR order.
    """
    frame = imread(file_path)
    if frame is None:
        raise ValueError(f"Cannot read {file_path}")
    return frame


def predict_yolo_frames(
    model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
) -> tuple[pl.DataFrame, float]:
    """Predict every frame of an image table with a trained YOLO checkpoint, one at a time.

    The time per image covers preprocessing, the forward pass, and NMS, with the GPU
    synchronized before and after; reading the file is not timed. `warmup_images` untimed
    passes on the first frame come first. Options follow
    https://docs.ultralytics.com/modes/predict/#inference-arguments

    Args:
        model: Detector from `build_model`; unused, since the checkpoint holds the whole model.
        checkpoint_path: Path of the run's `best.pt`.
        images: Image table with `image_id` and absolute `file_path`.
        config: Resolved run configuration with `imgsz` and `[inference]` `conf`, `iou`,
            `max_det`, `agnostic_nms`, and `warmup_images`.

    Returns:
        Predictions with `image_id`, 0-based `class_id`, `score`, and pixel `x`, `y`, `w`,
        `h` (top-left corner), and the mean inference time in ms per image.
    """
    detector = load_yolo_checkpoint(checkpoint_path)
    inference = config["inference"]
    is_cuda = torch.cuda.is_available()

    def detect_boxes(frame: np.ndarray) -> Boxes:
        [result] = detector.predict(
            frame,
            imgsz=config["imgsz"],
            conf=inference["conf"],
            iou=inference["iou"],
            max_det=inference["max_det"],
            agnostic_nms=inference["agnostic_nms"],
            device=0 if is_cuda else "cpu",
            verbose=False,
        )
        if not isinstance(result, Results) or result.boxes is None:
            raise TypeError(f"Expected detection results with boxes, got {result!r}")
        return result.boxes

    file_paths = images["file_path"].to_list()
    warmup_frame = read_bgr_frame(file_paths[0])
    for _ in range(inference["warmup_images"]):
        detect_boxes(warmup_frame)
    frame_detections, seconds = [], []
    for file_path in tqdm(file_paths, desc="Predicting frames", unit="frame"):
        frame = read_bgr_frame(file_path)
        if is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        boxes = detect_boxes(frame)
        if is_cuda:
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        frame_detections.append(convert_yolo_boxes(boxes))
    detections = np.concatenate(frame_detections)
    frame_index = np.repeat(np.arange(len(file_paths)), [len(d) for d in frame_detections])
    predictions = pl.DataFrame(
        {
            "image_id": images["image_id"].gather(frame_index),
            "class_id": detections[:, 0].astype(np.int64),
            "score": detections[:, 1],
            **{column: detections[:, 2 + i] for i, column in enumerate(BOX_COLUMNS)},
        }
    )
    return predictions, 1000 * float(np.mean(seconds))
