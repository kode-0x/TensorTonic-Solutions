import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)

    result = np.zeros_like(X, dtype=float)
    np.divide(X - mean, std, out=result, where=std > eps)

    return result
