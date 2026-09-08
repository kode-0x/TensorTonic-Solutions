import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.asarray(x)
    pmf = np.where(x == 1, p, 1 - p)
    mean = float(p)
    var = float(p * (1 - p))
    return {
        "pmf": pmf,
        "mean": mean,
        "variance": var
    }
