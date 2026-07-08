def calculate_character_accuracy(prediction: str, ground_truth: str) -> float:
    """Calculate character-level accuracy."""
    if not ground_truth:
        return 1.0 if not prediction else 0.0
    
    correct = sum(1 for p, g in zip(prediction, ground_truth) if p == g)
    return correct / len(ground_truth)


def calculate_character_accuracy_batch(
    predictions: list,
    ground_truths: list
) -> dict:
    """Calculate character accuracy for batch."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    acc_values = [
        calculate_character_accuracy(p, g)
        for p, g in zip(predictions, ground_truths)
    ]
    return {
        "character_accuracy_mean": sum(acc_values) / len(acc_values) if acc_values else 0.0,
        "character_accuracy_min": min(acc_values) if acc_values else 0.0,
        "character_accuracy_max": max(acc_values) if acc_values else 0.0,
    }
