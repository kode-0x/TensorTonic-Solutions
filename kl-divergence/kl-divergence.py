import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = p > 0
    q = np.maximum(q[mask], eps)

    return float(np.sum(p[mask] * np.log(p[mask] / q)))