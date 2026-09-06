import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(-np.mean(np.log(y_pred[np.arange(len(y_true)), y_true])))