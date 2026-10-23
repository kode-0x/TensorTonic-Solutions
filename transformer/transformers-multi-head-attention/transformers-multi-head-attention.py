import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    W_k = np.asarray(W_k, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)

    batch, query_len, d_model = Q.shape
    key_len = K.shape[1]
    head_dim = d_model // num_heads

    Q_proj = (Q @ W_q).reshape(batch, query_len, num_heads, head_dim)
    K_proj = (K @ W_k).reshape(batch, key_len, num_heads, head_dim)
    V_proj = (V @ W_v).reshape(batch, V.shape[1], num_heads, head_dim)

    Q_proj = Q_proj.transpose(0, 2, 1, 3)
    K_proj = K_proj.transpose(0, 2, 1, 3)
    V_proj = V_proj.transpose(0, 2, 1, 3)

    scores = (Q_proj @ K_proj.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores -= np.max(scores, axis=-1, keepdims=True)

    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    heads = weights @ V_proj
    heads = heads.transpose(0, 2, 1, 3).reshape(batch, query_len, d_model)

    return heads @ W_o