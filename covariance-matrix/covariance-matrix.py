import numpy as np

def covariance_matrix(X):
    """
    Compute the sample covariance matrix without using np.cov.
    """
    X = np.asarray(X)

    if X.ndim != 2 or X.shape[0] < 2:
        return None

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    return (X_centered.T @ X_centered) / (X.shape[0] - 1)
