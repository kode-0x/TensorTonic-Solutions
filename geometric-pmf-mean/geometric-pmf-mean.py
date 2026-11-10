
import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with PMF and mean.
    """
    k = np.array(k)

    pmf = (1 - p) ** (k - 1) * p
    mean = float(1 / p)

    return {
        "pmf": pmf,
        "mean": mean
    }
