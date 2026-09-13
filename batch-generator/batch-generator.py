import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        end = start + batch_size

        if drop_last and end > len(X):
            break

        idx = indices[start:end]
        yield X[idx], y[idx]
