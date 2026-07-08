import editdistance


def calculate_wer(prediction: str, ground_truth: str) -> float:
    """Calculate Word Error Rate."""
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    
    distance = editdistance.eval(pred_words, gt_words)
    return distance / len(gt_words)


def calculate_wer_batch(
    predictions: list,
    ground_truths: list
) -> dict:
    """Calculate WER for batch of predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    wer_values = [
        calculate_wer(p, g)
        for p, g in zip(predictions, ground_truths)
    ]
    return {
        "wer_mean": sum(wer_values) / len(wer_values) if wer_values else 0.0,
        "wer_min": min(wer_values) if wer_values else 0.0,
        "wer_max": max(wer_values) if wer_values else 0.0,
    }
