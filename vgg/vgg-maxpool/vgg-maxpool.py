import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    return np.maximum.reduce([
        x[:, 0::2, 0::2, :],
        x[:, 1::2, 0::2, :],
        x[:, 0::2, 1::2, :],
        x[:, 1::2, 1::2, :]
    ])
