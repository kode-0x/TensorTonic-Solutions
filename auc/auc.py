import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))