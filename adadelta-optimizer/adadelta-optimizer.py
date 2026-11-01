import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    """
    Returns a dictionary with new_w, new_E_grad_sq, and new_E_update_sq.
    """
    w = np.asarray(w)
    grad = np.asarray(grad)
    E_grad_sq = np.asarray(E_grad_sq)
    E_update_sq = np.asarray(E_update_sq)

    new_E_grad_sq = rho * E_grad_sq + (1 - rho) * grad**2

    delta_w = -np.sqrt(E_update_sq + eps) / np.sqrt(new_E_grad_sq + eps) * grad

    new_E_update_sq = rho * E_update_sq + (1 - rho) * delta_w**2

    new_w = w + delta_w

    return {
        "new_w": new_w,
        "new_E_grad_sq": new_E_grad_sq,
        "new_E_update_sq": new_E_update_sq
    }