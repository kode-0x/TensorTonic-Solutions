import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    g = np.asarray(x, dtype=np.float64).copy()

    for J in gradients_F:
        J = np.asarray(J, dtype=np.float64)
        g = g @ (np.eye(J.shape[0]) + J)

    return g


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    g = np.asarray(x, dtype=np.float64).copy()

    for J in gradients_F:
        J = np.asarray(J, dtype=np.float64)
        g = g @ J

    return g