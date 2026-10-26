import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    B, H, W, C = image.shape
    h, w = H // patch_size, W // patch_size

    patches = image[:, :h * patch_size, :w * patch_size, :]
    patches = patches.reshape(B, h, patch_size, w, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, h * w, patch_size * patch_size * C)

    return (patches @ W_proj + bias).astype(np.float64)