import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    total = 0.0

    for p, y in zip(predictions, targets):
        pt = p if y == 1 else 1 - p
        loss = -alpha * ((1 - pt) ** gamma) * math.log(pt)
        total += loss

    return total / len(predictions)