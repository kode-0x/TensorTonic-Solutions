import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g = np.asarray(g, dtype=float)

    norm = np.linalg.norm(g)

    if norm > max_norm:
        g = g * (max_norm / norm)

    return g