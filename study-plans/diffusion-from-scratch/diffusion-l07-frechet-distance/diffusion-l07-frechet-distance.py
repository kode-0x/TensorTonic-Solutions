import numpy as np

def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Returns the squared Frechet distance between Gaussian fits of the two feature sets as a Python float.
    """
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)

    cov_a = np.atleast_2d(np.cov(features_a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(features_b, rowvar=False))

    eigvals_a, eigvecs_a = np.linalg.eigh(cov_a)
    eigvals_a = np.maximum(eigvals_a, 0.0)

    sqrt_a = eigvecs_a @ np.diag(np.sqrt(eigvals_a)) @ eigvecs_a.T

    middle = sqrt_a @ cov_b @ sqrt_a
    middle = (middle + middle.T) / 2.0

    eigvals = np.linalg.eigvalsh(middle)
    eigvals = np.maximum(eigvals, 0.0)

    mean_term = np.sum((mu_a - mu_b) ** 2)
    trace_term = np.trace(cov_a) + np.trace(cov_b)
    cross_term = 2.0 * np.sum(np.sqrt(eigvals))

    return float(mean_term + trace_term - cross_term)