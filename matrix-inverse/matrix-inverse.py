import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    aug = np.concatenate((A, np.eye(n)), axis=1)

    for col in range(n):
        pivot = col + np.argmax(np.abs(aug[col:, col]))

        if abs(aug[pivot, col]) < 1e-12:
            return None

        aug[[col, pivot]] = aug[[pivot, col]]

        aug[col] /= aug[col, col]

        for row in range(n):
            if row != col:
                aug[row] -= aug[row, col] * aug[col]

    return aug[:, n:]
