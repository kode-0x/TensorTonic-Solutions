import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x)

    random_values = (
        rng.random(x.shape)
        if rng is not None
        else np.random.random(x.shape)
    )

    scale = 1 / (1 - p)
    dropout_pattern = (random_values >= p) * scale
    output = x * dropout_pattern

    return output, dropout_pattern