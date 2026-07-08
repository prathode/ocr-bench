from typing import Tuple
import numpy as np


def calculate_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """Calculate Intersection over Union for two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def calculate_iou_batch(
    predictions: list,
    ground_truths: list
) -> dict:
    """Calculate IoU statistics for batch."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    ious = [
        calculate_iou(p, g)
        for p, g in zip(predictions, ground_truths)
    ]
    return {
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "iou_std": float(np.std(ious)) if ious else 0.0,
        "iou_min": float(min(ious)) if ious else 0.0,
        "iou_max": float(max(ious)) if ious else 0.0,
    }
