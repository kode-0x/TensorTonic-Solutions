import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    X = hidden_states[:, 0, :]
    B = X.shape[0]

    logits = X @ classifier_W + classifier_b
    logits -= np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = float(-np.mean(np.log(probs[np.arange(B), labels])))

    probs[np.arange(B), labels] -= 1
    probs /= B

    grad_W = X.T @ probs
    grad_b = np.sum(probs, axis=0)

    new_classifier_W = classifier_W - learning_rate * grad_W
    new_classifier_b = classifier_b - learning_rate * grad_b

    return {
        "new_classifier_W": new_classifier_W.astype(np.float64),
        "new_classifier_b": new_classifier_b.astype(np.float64),
        "loss": loss
    }