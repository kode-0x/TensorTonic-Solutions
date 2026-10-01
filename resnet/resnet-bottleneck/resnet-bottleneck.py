import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    W3 = np.asarray(W3, dtype=np.float64)

    out = np.maximum(0, x @ W1)
    out = np.maximum(0, out @ W2)
    out = out @ W3

    if Ws is None:
        shortcut = x
    else:
        Ws = np.asarray(Ws, dtype=np.float64)
        shortcut = x @ Ws

    out = np.maximum(0, out + shortcut)

    return np.round(out, 4).tolist()