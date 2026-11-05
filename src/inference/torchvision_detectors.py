import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torchvision.io import ImageReadMode, decode_image
from tqdm.auto import tqdm

from src.inference.detections import build_prediction_table
from src.models.torchvision_labels import shift_detections_from_torchvision

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
    model: torch.nn.Module, images: pl.DataFrame, device: torch.device, warmup_images: int
) -> tuple[pl.DataFrame, float]:
    """Detect objects on every frame of an image table with a torchvision detector, batch 1.

    The model is put in eval mode on `device`. The time per image covers moving the frame to
    the device, the uint8 to float conversion, and the forward pass with resizing and NMS,
    with the GPU synchronized before and after; decoding the file is not timed, as for
    YOLO. `warmup_images` untimed passes on the first frame come first.

    Args:
        model: torchvision detector with background label 0, already on `device`.
        images: Image table with `image_id` and absolute `file_path`.
        device: Device of the model.
        warmup_images: Number of untimed passes before measuring.

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
            [detections] = shift_detections_from_torchvision(outputs)
            frame_detections.append(convert_torchvision_detections(detections))
    predictions = build_prediction_table(images["image_id"], frame_detections)
    return predictions, 1000 * float(np.mean(seconds))


def predict_torchvision_frames(
    model: torch.nn.Module, checkpoint_path: Path, images: pl.DataFrame, config: dict
) -> tuple[pl.DataFrame, float]:
    """Predict every frame of an image table with a trained torchvision checkpoint.

    The score threshold, NMS IoU, and detections per frame of `[inference]` were set in the
    model by its builder.

    Args:
        model: Detector from `build_model` with the run's configuration.
        checkpoint_path: Path of a checkpoint written by the torchvision training loop.
        images: Image table with `image_id` and absolute `file_path`.
        config: Resolved run configuration with `[inference]` `warmup_images`.

    Returns:
        Predictions with `image_id`, 0-based `class_id`, `score`, and pixel `x`, `y`, `w`,
        `h` (top-left corner), and the mean inference time in ms per image.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint[MODEL_STATE_KEY])
    device = select_device()
    model.to(device)
    return detect_frames(model, images, device, config["inference"]["warmup_images"])
