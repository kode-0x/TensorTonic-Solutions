import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.asarray(y)

    if num_classes is None:
        num_classes = np.max(y) + 1

    result = np.zeros((len(y), num_classes), dtype=float)
    result[np.arange(len(y)), y] = 1.0

    return result
