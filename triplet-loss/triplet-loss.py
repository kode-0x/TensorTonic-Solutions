import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)

    if anchor.ndim == 1:
        d_positive = np.sum((anchor - positive) ** 2)
        d_negative = np.sum((anchor - negative) ** 2)
    else:
        d_positive = np.sum((anchor - positive) ** 2, axis=1)
        d_negative = np.sum((anchor - negative) ** 2, axis=1)

    loss = np.maximum(0, d_positive - d_negative + margin)

    return float(np.mean(loss))
