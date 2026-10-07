import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    p = np.asarray(p)
    y = np.asarray(y)

    loss = -(y * (1 - p) ** gamma * np.log(p) + (1 - y) * p ** gamma * np.log(1 - p))

    return float(np.mean(loss))