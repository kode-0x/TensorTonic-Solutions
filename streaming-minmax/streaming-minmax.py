import numpy as np

def streaming_minmax(D: int, batches: list, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with normalized_batches, min, and max.
    """
    running_min = np.full(D, np.inf)
    running_max = np.full(D, -np.inf)

    normalized_batches = []

    for batch in batches:
        batch = np.asarray(batch)

        running_min = np.minimum(running_min, np.min(batch, axis=0))
        running_max = np.maximum(running_max, np.max(batch, axis=0))

        normalized = (batch - running_min) / np.maximum(running_max - running_min, eps)

        normalized_batches.append(normalized)

    return {
        "normalized_batches": normalized_batches,
        "min": running_min,
        "max": running_max
    }