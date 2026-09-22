import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))