import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)

    eps = 1e-5

    def bn(z, gamma, beta):
        mean = np.mean(z, axis=0)
        var = np.var(z, axis=0)
        z = (z - mean) / np.sqrt(var + eps)
        return gamma * z + beta

    if mode == "post":
        out = x @ W1
        out = bn(out, gamma1, beta1)
        out = np.maximum(0, out)
        out = out @ W2
        out = bn(out, gamma2, beta2)
        out = np.maximum(0, out + x)
    else:
        out = bn(x, gamma1, beta1)
        out = np.maximum(0, out)
        out = out @ W1
        out = bn(out, gamma2, beta2)
        out = np.maximum(0, out)
        out = out @ W2
        out = out + x

    return {
        "output": np.round(out, 4).tolist(),
        "mode": mode
    }