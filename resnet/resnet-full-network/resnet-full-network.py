import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    conv1 = np.asarray(conv1, dtype=np.float64)
    W1_b1 = np.asarray(W1_b1, dtype=np.float64)
    W2_b1 = np.asarray(W2_b1, dtype=np.float64)
    W1_b2 = np.asarray(W1_b2, dtype=np.float64)
    W2_b2 = np.asarray(W2_b2, dtype=np.float64)
    Ws_b2 = np.asarray(Ws_b2, dtype=np.float64)
    fc = np.asarray(fc, dtype=np.float64)

    relu = lambda z: np.maximum(0, z)

    out = relu(x @ conv1)

    residual = out
    out = relu(out @ W1_b1)
    out = out @ W2_b1
    out = relu(out + residual)

    residual = out @ Ws_b2
    out = relu(out @ W1_b2)
    out = out @ W2_b2
    out = relu(out + residual)

    logits = out @ fc

    return np.round(logits, 4).tolist()
