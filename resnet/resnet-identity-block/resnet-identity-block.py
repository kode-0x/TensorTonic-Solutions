import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)

    out = np.maximum(0, x @ W1.T)
    out = out @ W2.T
    out = np.maximum(0, out + x)

    return np.round(out, 4).tolist()