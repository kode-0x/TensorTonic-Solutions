import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    w = np.asarray(w)
    v = np.asarray(v)
    grad = np.asarray(grad)

    new_v = momentum * v + lr * grad
    new_w = w - new_v

    return {"new_w": new_w, "new_v": new_v}