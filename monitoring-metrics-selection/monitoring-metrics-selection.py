import numpy as np

def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if system_type == "classification":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    elif system_type == "regression":
        errors = y_true - y_pred

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        return {
            "mae": float(mae),
            "rmse": float(rmse)
        }

    elif system_type == "ranking":
        order = np.argsort(-y_pred, kind="stable")
        top_3 = y_true[order[:3]]

        relevant = np.sum(y_true > 0)
        relevant_at_3 = np.sum(top_3 > 0)

        precision_at_3 = relevant_at_3 / 3
        recall_at_3 = relevant_at_3 / relevant if relevant else 0.0

        return {
            "precision_at_3": float(precision_at_3),
            "recall_at_3": float(recall_at_3)
        }