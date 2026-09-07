import numpy as np

def euclidean_distance(x: list[float], y: list[float]) -> float:
    """
    Returns the Euclidean (L2) distance between two vectors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(np.sqrt(np.sum((x - y) ** 2)))