import numpy as np

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    X = np.asarray(X, dtype=np.float64)
    N, T, D = X.shape
    H = W_f.shape[0]
    O = W_y.shape[0]

    h = np.zeros((N, H), dtype=np.float64)
    c = np.zeros((N, H), dtype=np.float64)
    outputs = np.zeros((N, T, O), dtype=np.float64)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

    for t in range(T):
        x_t = X[:, t, :]
        combined = np.concatenate((h, x_t), axis=1)

        f = sigmoid(combined @ W_f.T + b_f)
        i = sigmoid(combined @ W_i.T + b_i)
        c_candidate = np.tanh(combined @ W_c.T + b_c)
        o = sigmoid(combined @ W_o.T + b_o)

        c = f * c + i * c_candidate
        h = o * np.tanh(c)

        outputs[:, t, :] = h @ W_y.T + b_y

    return {
        "outputs": outputs,
        "final_hidden_state": h,
        "final_cell_state": c
    }