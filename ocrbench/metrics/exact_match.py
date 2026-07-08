def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match accuracy (1.0 or 0.0)."""
    return 1.0 if prediction == ground_truth else 0.0


def calculate_exact_match_batch(
    predictions: list,
    ground_truths: list
) -> dict:
    """Calculate exact match for batch."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    matches = [
        1.0 if p == g else 0.0
        for p, g in zip(predictions, ground_truths)
    ]
    return {
        "exact_match_accuracy": sum(matches) / len(matches) if matches else 0.0,
        "total_matches": sum(matches),
        "total_samples": len(matches),
    }
