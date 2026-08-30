import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)

    angle = pos / np.power(base, i / d_model)

    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, :pe[:, 1::2].shape[1]])

    return pe