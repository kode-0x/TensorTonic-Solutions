import math
import numpy as np

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    W = np.asarray(W, dtype=float)

    L = math.sqrt(6 / fan_in)
    W = W * (2 * L) - L

    return W.tolist()