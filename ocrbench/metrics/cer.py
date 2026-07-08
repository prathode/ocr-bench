import editdistance


def calculate_cer(prediction: str, ground_truth: str) -> float:
    """Calculate Character Error Rate."""
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    distance = editdistance.eval(prediction, ground_truth)
    return distance / len(ground_truth)


def calculate_cer_batch(
    predictions: list,
    ground_truths: list
) -> dict:
    """Calculate CER for batch of predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    cer_values = [
        calculate_cer(p, g)
        for p, g in zip(predictions, ground_truths)
    ]
    return {
        "cer_mean": sum(cer_values) / len(cer_values) if cer_values else 0.0,
        "cer_min": min(cer_values) if cer_values else 0.0,
        "cer_max": max(cer_values) if cer_values else 0.0,
    }
