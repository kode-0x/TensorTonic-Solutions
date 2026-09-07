import numpy as np

def manhattan_distance(x: list[float], y: list[float]) -> float:
    """
    Returns the Manhattan (L1) distance between two vectors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(np.abs(x - y).sum())