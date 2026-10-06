import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the contrastive loss as a float.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)

    if a.ndim == 1:
        d = np.sqrt(np.sum((a - b) ** 2))
    else:
        d = np.sqrt(np.sum((a - b) ** 2, axis=1))

    loss = y * d**2 + (1 - y) * np.maximum(0, margin - d)**2

    if reduction == "mean":
        return float(np.mean(loss))
    else:
        return float(np.sum(loss))
