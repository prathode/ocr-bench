from typing import List, Tuple
import numpy as np

from .iou import calculate_iou


def calculate_detection_accuracy(
    predictions: List[List[Tuple[int, int, int, int]]],
    ground_truths: List[List[Tuple[int, int, int, int]]],
    iou_threshold: float = 0.5
) -> dict:
    """Calculate detection accuracy based on IoU threshold."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = 0
    total = len(predictions)

    for pred_bboxes, gt_bboxes in zip(predictions, ground_truths):
        if not gt_bboxes:
            continue
        
        for gt_bbox in gt_bboxes:
            max_iou = 0.0
            for pred_bbox in pred_bboxes:
                iou = calculate_iou(pred_bbox, gt_bbox)
                max_iou = max(max_iou, iou)
            
            if max_iou >= iou_threshold:
                correct += 1

    return {
        "detection_accuracy": correct / total if total > 0 else 0.0,
        "correct_detections": correct,
        "total_ground_truths": total,
    }
