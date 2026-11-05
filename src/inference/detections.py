import numpy as np
import polars as pl

from src.data.annotations import BOX_COLUMNS


def build_prediction_table(
    image_ids: pl.Series, frame_detections: list[np.ndarray]
) -> pl.DataFrame:
    """Combine the detections of each frame into one prediction table.

    Args:
        image_ids: Image id of each frame, in the order of `frame_detections`.
        frame_detections: Per frame, an array of shape (n, 6) with the 0-based class id,
            score, and `x, y, w, h` in px (top-left corner).

    Returns:
        Predictions with `image_id`, int64 `class_id`, `score`, and pixel `x`, `y`, `w`, `h`.
    """
    detections = np.concatenate(frame_detections)
    frame_index = np.repeat(np.arange(len(frame_detections)), [len(d) for d in frame_detections])
    return pl.DataFrame(
        {
            "image_id": image_ids.gather(frame_index),
            "class_id": detections[:, 0].astype(np.int64),
            "score": detections[:, 1],
            **{column: detections[:, 2 + i] for i, column in enumerate(BOX_COLUMNS)},
        }
    )
