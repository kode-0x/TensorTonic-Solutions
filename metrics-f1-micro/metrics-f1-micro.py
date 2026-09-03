def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum(a == b for a, b in zip(y_true, y_pred))
    fp = len(y_pred) - tp
    fn = len(y_true) - tp

    f1 = (2 * tp) / (2 * tp + fp + fn)

    return round(float(f1), 4)