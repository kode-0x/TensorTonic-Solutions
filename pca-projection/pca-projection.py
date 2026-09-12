import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X, dtype=float)

    Xc = X - np.mean(X, axis=0)

    cov = (Xc.T @ Xc) / (X.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1][:k]
    W = eigenvectors[:, idx]

    result = Xc @ W

    return result.tolist()
