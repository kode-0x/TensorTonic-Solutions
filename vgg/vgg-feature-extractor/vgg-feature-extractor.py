import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    out = x
    weight_idx = 0

    for layer in config:
        if layer == 'M':
            out = maxpool_2x2(out)
        else:
            out = out @ conv_weights[weight_idx] + conv_biases[weight_idx]
            out = np.maximum(out, 0)
            weight_idx += 1

    return out