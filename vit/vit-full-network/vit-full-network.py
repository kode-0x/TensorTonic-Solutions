import numpy as np

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    B, H, W, C = image.shape
    p = patch_size
    h, w = H // p, W // p

    patches = image[:, :h * p, :w * p, :].reshape(B, h, p, w, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, h * w, p * p * C)
    tokens = patches @ W_patch + patch_bias
    tokens = np.concatenate([np.broadcast_to(cls_token, (B, 1, cls_token.shape[-1])), tokens], axis=1)
    tokens = tokens + pos_embed

    def layer_norm(x):
        return (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-6)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    for weights in encoder_weights:
        Wq, Wk, Wv, Wo, W1, W2 = (weights[k] for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"))
        B, N, D = tokens.shape
        dh = D // num_heads
        xn = layer_norm(tokens)
        Q = (xn @ Wq).reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
        K = (xn @ Wk).reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
        V = (xn @ Wv).reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn /= np.sum(attn, axis=-1, keepdims=True)
        A = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
        y = tokens + A @ Wo
        tokens = y + gelu(layer_norm(y) @ W1) @ W2

    return (layer_norm(tokens[:, 0, :]) @ W_head).astype(np.float64)