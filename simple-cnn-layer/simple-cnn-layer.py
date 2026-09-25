import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    N, Cin, H, W_in = x.shape
    Cout, _, KH, KW = W.shape

    OH = H - KH + 1
    OW = W_in - KW + 1

    out = np.empty((N, Cout, OH, OW), dtype=float)

    for n in range(N):
        for c in range(Cout):
            for i in range(OH):
                for j in range(OW):
                    out[n, c, i, j] = np.sum(
                        x[n, :, i:i + KH, j:j + KW] * W[c]
                    ) + b[c]

    return out
