import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)

    sigmoid = lambda v: 1 / (1 + np.exp(-v))

    z = sigmoid(x @ params["Wz"] + h_prev @ params["Uz"] + params["bz"])
    r = sigmoid(x @ params["Wr"] + h_prev @ params["Ur"] + params["br"])

    h_candidate = np.tanh(
        x @ params["Wh"] +
        (r * h_prev) @ params["Uh"] +
        params["bh"]
    )

    return (1 - z) * h_prev + z * h_candidate