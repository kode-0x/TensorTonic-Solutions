import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    X = np.array(X)
    y = np.array(y)

    XT = X.T
    w = np.linalg.inv(XT @ X) @ XT @ y

    return w.tolist()
