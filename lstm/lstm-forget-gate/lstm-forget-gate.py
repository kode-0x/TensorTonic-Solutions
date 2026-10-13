import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Returns the float64 forget-gate values.
    """
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    W_f = np.asarray(W_f, dtype=np.float64)
    b_f = np.asarray(b_f, dtype=np.float64)

    combined = np.concatenate((h_prev, x_t), axis=-1)
    logits = combined @ W_f.T + b_f
    return np.asarray(sigmoid(logits), dtype=np.float64)