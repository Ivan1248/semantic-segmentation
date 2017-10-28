from processing import to_laplacian_pyramid, normalize, rgb_to_yuv, transform as transf
import numpy as np
import random


def random_transform(image: np.ndarray, labeling: np.ndarray, max_rot=8, max_scale_delta=0.1) -> tuple:  # TODO 1
    rot = random.uniform(-max_rot, max_rot)
    zoom = 1 + random.uniform(-max_scale_delta, max_scale_delta)
    flip = bool(random.getrandbits(1))
    return transf(image, rot, zoom, flip), transf(labeling, rot, zoom, flip)
