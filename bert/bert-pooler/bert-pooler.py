import numpy as np

def bert_pooler(hidden_states: np.ndarray, W_pool: np.ndarray,
                b_pool: np.ndarray) -> np.ndarray:
    """
    Returns the float64 pooled states with shape (B, D).
    """
    cls_states = hidden_states[:, 0, :]
    pooled = cls_states @ W_pool + b_pool
    return np.asarray(np.tanh(pooled), dtype=np.float64)