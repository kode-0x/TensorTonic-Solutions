import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    distances = np.linalg.norm(
        X[:, None, :] - X[None, :, :],
        axis=2
    )

    same_cluster = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_cluster, False)

    a = np.sum(distances * same_cluster, axis=1) / np.sum(same_cluster, axis=1)

    clusters = np.unique(labels)

    cluster_distances = np.array([
        np.mean(distances[:, labels == c], axis=1)
        for c in clusters
    ]).T

    cluster_distances[np.arange(len(X)), labels] = np.inf

    b = np.min(cluster_distances, axis=1)

    scores = (b - a) / np.maximum(a, b)

    return float(np.mean(scores))