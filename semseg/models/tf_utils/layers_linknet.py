import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from layers import *


def linknet_encoder(x,
                    is_training,
                    keep_shape,
                    reuse: bool = None,
                    scope: str = None):
    width = x.shape[-1] * (1 if keep_shape else 2)
    with tf.variable_scope(scope, 'LinkNetEncoder', reuse=reuse):
        x = residual_block(
            x,
            is_training,
            kind=ResidualBlockKind([3, 3]),
            width=width,
            force_downsampling=True)
        x = residual_block(
            x, is_training, kind=ResidualBlockKind([3, 3]), width=width)
    return x
