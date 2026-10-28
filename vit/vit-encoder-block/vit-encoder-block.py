import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    B, N, D = x.shape
    dh = D // num_heads

    def layer_norm(a):
        mean = np.mean(a, axis=-1, keepdims=True)
        var = np.var(a, axis=-1, keepdims=True)
        return (a - mean) / np.sqrt(var + 1e-6)

    def gelu(a):
        return 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a**3)))

    xn = layer_norm(x)

    Q = (xn @ Wq).reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
    K = (xn @ Wk).reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
    V = (xn @ Wv).reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)
    scores -= np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn /= np.sum(attn, axis=-1, keepdims=True)

    A = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
    y = x + A @ Wo
    z = y + gelu(layer_norm(y) @ W1) @ W2

    return z.astype(np.float64)