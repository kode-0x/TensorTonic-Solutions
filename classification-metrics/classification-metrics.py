import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))

    tp = np.array([np.sum((y_true == c) & (y_pred == c)) for c in classes])
    fp = np.array([np.sum((y_true != c) & (y_pred == c)) for c in classes])
    fn = np.array([np.sum((y_true == c) & (y_pred != c)) for c in classes])

    if average == "micro":
        t = np.sum(tp)
        f_p = np.sum(fp)
        f_n = np.sum(fn)

        precision = t / (t + f_p) if t + f_p else 0.0
        recall = t / (t + f_n) if t + f_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    elif average == "macro":
        precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall_c = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1_c = np.divide(
            2 * precision_c * recall_c,
            precision_c + recall_c,
            out=np.zeros_like(precision_c),
            where=(precision_c + recall_c) != 0
        )

        precision = np.mean(precision_c)
        recall = np.mean(recall_c)
        f1 = np.mean(f1_c)

    elif average == "weighted":
        precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall_c = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1_c = np.divide(
            2 * precision_c * recall_c,
            precision_c + recall_c,
            out=np.zeros_like(precision_c),
            where=(precision_c + recall_c) != 0
        )

        weights = np.array([np.sum(y_true == c) for c in classes], dtype=float)
        weights /= np.sum(weights)

        precision = np.sum(precision_c * weights)
        recall = np.sum(recall_c * weights)
        f1 = np.sum(f1_c * weights)

    else:
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)
    }