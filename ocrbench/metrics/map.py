import numpy as np

from .iou import calculate_iou


def calculate_map(
    predictions: list,
    ground_truths: list,
    iou_thresholds: list = None,
    class_thresholds: dict = None
) -> dict:
    """Calculate mean Average Precision."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    aps = []
    for iou_thr in iou_thresholds:
        tp = 0
        fp = 0
        
        for pred_bboxes, gt_bboxes in zip(predictions, ground_truths):
            for pred in pred_bboxes:
                is_tp = False
                for gt in gt_bboxes:
                    iou = calculate_iou(pred, gt)
                    if iou >= iou_thr:
                        is_tp = True
                        break
                tp += int(is_tp)
                fp += int(not is_tp)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        aps.append(precision)

    return {
        "map": float(np.mean(aps)) if aps else 0.0,
        "ap_per_threshold": dict(zip([str(t) for t in iou_thresholds], aps)),
    }
