import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2:
            return None

        if axis not in (0, 1, None):
            return None

        if norm_type not in ("l1", "l2", "max"):
            return None

        if norm_type == "l1":
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == "l2":
            norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        else:
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

        return np.divide(
            matrix,
            norm,
            out=np.zeros_like(matrix),
            where=norm != 0
        )

    except (ValueError, TypeError):
        return None
