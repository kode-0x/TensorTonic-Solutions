import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = 0.5 * error**2
    linear = delta * (abs_error - 0.5 * delta)

    loss = np.where(abs_error <= delta, quadratic, linear)

    return float(np.mean(loss))