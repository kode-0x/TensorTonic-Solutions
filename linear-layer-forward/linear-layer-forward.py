import numpy as np

def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)

    Y = X @ W + b

    return Y.tolist()