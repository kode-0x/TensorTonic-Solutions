import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    C_prev = np.asarray(C_prev, dtype=np.float64)

    combined = np.concatenate((h_prev, x_t), axis=-1)

    f_t = sigmoid(combined @ W_f.T + b_f)
    i_t = sigmoid(combined @ W_i.T + b_i)
    C_tilde = np.tanh(combined @ W_c.T + b_c)

    cell_state = f_t * C_prev + i_t * C_tilde

    o_t = sigmoid(combined @ W_o.T + b_o)
    hidden_state = o_t * np.tanh(cell_state)

    return {
        "hidden_state": np.asarray(hidden_state, dtype=np.float64),
        "cell_state": np.asarray(cell_state, dtype=np.float64)
    }