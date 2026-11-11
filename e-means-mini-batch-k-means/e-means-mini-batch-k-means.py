import numpy as np

def e_means(X: list, centroids: list, batches: list, alpha: float) -> dict:
    """
    Returns a dict with the final centroids (k, d), the running counts (k,), the labels (n,) and the mse.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float).copy()

    k, d = centroids.shape
    S = centroids.copy()
    N = np.ones(k, dtype=float)

    for batch in batches:
        points = X[batch]

        distances = np.sum(
            (points[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2
        )
        labels = np.argmin(distances, axis=1)

        s = np.zeros((k, d))
        n = np.zeros(k)

        for i, label in enumerate(labels):
            s[label] += points[i]
            n[label] += 1

        S = (1 - alpha) * S + alpha * s
        N = (1 - alpha) * N + alpha * n

        centroids = np.divide(
            S,
            N[:, None],
            out=centroids.copy(),
            where=N[:, None] != 0
        )

    distances = np.sum(
        (X[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2
    )

    labels = np.argmin(distances, axis=1)
    mse = float(np.mean(distances[np.arange(len(X)), labels]))

    return {
        "centroids": centroids,
        "counts": N,
        "labels": labels,
        "mse": mse
    }