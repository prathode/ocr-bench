from typing import List, Tuple


def calculate_precision_recall_f1(
    predictions: List[str],
    ground_truths: List[str]
) -> dict:
    """Calculate precision, recall, F1 for OCR predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_chars = set(pred)
        gt_chars = set(gt)
        
        tp = len(pred_chars & gt_chars)
        fp = len(pred_chars - gt_chars)
        fn = len(gt_chars - pred_chars)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
