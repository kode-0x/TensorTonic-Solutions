import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T)
    points = np.asarray(points)

    single = points.ndim == 1

    if single:
        points = points.reshape(1, 3)

    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1))],
        axis=1
    )

    transformed = homogeneous @ T.T

    result = transformed[:, :3]

    return result[0] if single else result