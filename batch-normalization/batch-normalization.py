import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta

    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.var(x, axis=(0, 2, 3), keepdims=True)

    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)

    normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * normalized + beta