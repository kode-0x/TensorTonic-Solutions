import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    positions = np.arange(seq_length, dtype=np.float64)[:, np.newaxis]
    dimensions = np.arange(0, d_model, 2, dtype=np.float64)
    angles = positions / np.power(10000.0, dimensions / d_model)

    encoding = np.zeros((seq_length, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)

    return encoding