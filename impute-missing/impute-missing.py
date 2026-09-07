import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    X = np.asarray(X, dtype=float)
    result = X.copy()

    if X.ndim == 1:
        values = np.nanmean(X) if strategy == "mean" else np.nanmedian(X)
        if np.isnan(values):
            values = 0.0
        result[np.isnan(result)] = values

    else:
        values = (
            np.nanmean(X, axis=0)
            if strategy == "mean"
            else np.nanmedian(X, axis=0)
        )
        values = np.where(np.isnan(values), 0.0, values)

        rows, cols = np.where(np.isnan(result))
        result[rows, cols] = values[cols]

    return result