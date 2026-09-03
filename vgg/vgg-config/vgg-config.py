import numpy as np

def make_vgg_config(variant: str) -> list:
    configs = {
        "vgg11": [1, 1, 2, 2, 2],
        "vgg13": [2, 2, 2, 2, 2],
        "vgg16": [2, 2, 3, 3, 3],
        "vgg19": [2, 2, 4, 4, 4]
    }

    blocks = configs[variant.lower()]
    channels = [64, 128, 256, 512, 512]

    config = []

    for count, channel in zip(blocks, channels):
        config.extend([channel] * count)
        config.append("M")

    return config