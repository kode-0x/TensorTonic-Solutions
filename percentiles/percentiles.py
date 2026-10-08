import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x = np.asarray(x)
    q = np.asarray(q)

    x = np.sort(x)

    r = (q / 100.0) * (x.size - 1)

    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)

    w = r - l

    return (1 - w) * x[l] + w * x[u]