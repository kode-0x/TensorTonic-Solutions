import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    nl = len(y_left)
    nr = len(y_right)
    n = nl + nr

    if n == 0:
        return 0.0

    return float((nl * gini(y_left) + nr * gini(y_right)) / n)