import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    p = np.asarray(p)
    y = np.asarray(y)

    dice = (2 * np.sum(p * y) + eps) / (np.sum(p) + np.sum(y) + eps)
    loss = 1 - dice

    return float(loss)
