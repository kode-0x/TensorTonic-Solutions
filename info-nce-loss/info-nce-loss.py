import numpy as np

def info_nce_loss(
    Z1: list[list[float]],
    Z2: list[list[float]],
    temperature: float
) -> float:
    """
    Returns the one-directional InfoNCE loss.
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    logits = (Z1 @ Z2.T) / temperature

    shifted = logits - np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(shifted)
    log_probs = shifted - np.log(np.sum(exp_logits, axis=1))

    loss = -np.diag(log_probs)

    return float(np.mean(loss))