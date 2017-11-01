from typing import List
import numpy as np

# rgb2yuv in skimage.color

def normalize(image: np.ndarray) -> np.ndarray:
    """ Converts values from [0, 255] to [0.0, 1.0]. """
    return image.astype(np.float32) * (1.0 / 255)


def denormalize(image: np.ndarray) -> np.ndarray:
    """ Converts values from [0.0, 1.0] to [0. 255]. """
    return (image * 255).astype(np.uint8)
