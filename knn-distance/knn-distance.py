import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 1:
        X_test = X_test[:, np.newaxis]

    distances = np.sqrt(
        np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)
    )

    indices = np.argsort(distances, axis=1)

    n_train = X_train.shape[0]
    result = np.full((X_test.shape[0], k), -1, dtype=int)
    result[:, :min(k, n_train)] = indices[:, :min(k, n_train)]

    return result
