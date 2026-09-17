import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    losses = []

    for y, p in zip(y_true, y_pred):
        p = min(1 - eps, max(eps, p))
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        losses.append(float(loss))

    return losses