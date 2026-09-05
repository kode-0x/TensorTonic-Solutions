from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    freq = Counter(x)
    mode = min(freq, key=lambda v: (-freq[v], v))

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode)
    }