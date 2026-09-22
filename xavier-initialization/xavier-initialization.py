import math
import numpy as np

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    W = np.asarray(W, dtype=float)

    L = math.sqrt(6 / (fan_in + fan_out))
    W = W * (2 * L) - L

    return W.tolist()