import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))

    means = x[indices].mean(axis=1)

    alpha = (1 - ci) / 2

    return {
        "bootstrap_mean": float(np.mean(means)),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha))
    }
