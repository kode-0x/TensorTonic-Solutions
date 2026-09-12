import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.asarray(x)

    n = len(x)
    mean = np.mean(x)
    s = np.sqrt(np.sum((x - mean) ** 2) / (n - 1))

    return float((mean - mu0) / (s / np.sqrt(n)))
