import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    return (np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y).tolist()
