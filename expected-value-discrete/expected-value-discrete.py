import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)

    if x.shape != p.shape:
        raise ValueError("x And p Must Have The Same Shape")

    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities Must Sum To 1")

    return float(np.sum(x * p))