import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    for W, b in zip(weights, biases):
        x = np.matmul(x, W) + b
        x = np.maximum(x, 0)

    return x
