import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    return np.mean((y_pred - y_true) ** 2).astype(float)