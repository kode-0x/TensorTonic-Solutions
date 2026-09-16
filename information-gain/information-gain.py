import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.array(y)
    split_mask = np.array(split_mask)

    def entropy(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return float(-np.sum(p * np.log2(p)))

    left = y[split_mask]
    right = y[~split_mask]

    if len(left) == 0 or len(right) == 0:
        return 0.0

    n = len(y)

    return float(
        entropy(y)
        - (len(left) / n) * entropy(left)
        - (len(right) / n) * entropy(right)
    )