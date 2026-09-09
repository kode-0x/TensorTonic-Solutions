import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score, kind="stable")
    y_true = y_true[order]
    y_score = y_score[order]

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    distinct = np.r_[np.where(np.diff(y_score) != 0)[0], len(y_score) - 1]

    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = y_score[distinct]

    tpr = tp / positives
    fpr = fp / negatives

    return {
        "fpr": np.r_[0.0, fpr],
        "tpr": np.r_[0.0, tpr],
        "thresholds": np.r_[np.inf, thresholds]
    }
