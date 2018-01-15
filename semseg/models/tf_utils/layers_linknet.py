import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from layers import *

from semseg.models.tf_utils.layers import conv, conv_transp, residual_block, ResidualBlockKind, bn_relu, max_pool


def linknet_encoder(x, is_training, width, reuse: bool = None, scope: str = None):
    with tf.variable_scope(scope, 'LinkNetEncoder', reuse=reuse):
        x = residual_block(
            x, is_training, kind=ResidualBlockKind([3, 3]), width=width)
        x = residual_block(
            x, is_training, kind=ResidualBlockKind([3, 3]), width=width)
    return x

def linknet_decoder(x, is_training = None, width = 32, reuse: bool = None, scope: str = None):
    N, H, W, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    m = int(x.shape[-1])
    with tf.variable_scope(scope, 'LinkNetDecoder', reuse=reuse):
        h = conv(
            x, ksize=1, width=int(m/4))
        h = bn_relu(h, is_training)
        h = conv_transp(h, ksize=3, stride=2, width=m//4)
        h = bn_relu(h, is_training)
        h = conv(h, ksize=1, width=width)
        h = bn_relu(h, is_training)
    return h

def linknet(x,
           is_training,
           reuse: bool = None,
           scope: str = None):
    encoder_widths = [64, 128, 256, 512]
    decoder_widths = [64, 64, 128, 256]
    encoder_outputs = []

    h = conv(x, ksize=7, width=64, stride=2)
    h = bn_relu(h, is_training)
    h = max_pool(h, stride=2, ksize=3)

    for i in range(4):
        h = linknet_encoder(h, is_training, encoder_widths[i], reuse, scope)
        encoder_outputs.append(h)

    for i in range(1, 4):
        h = linknet_decoder(h, is_training, decoder_widths[-i], reuse, scope)
        h = h + encoder_outputs[-(i+1)]

    h = linknet_decoder(h, is_training, decoder_widths[-4], reuse, scope)
    h = conv_transp(h, ksize=3, width=32, stride=2, reuse=reuse, scope=scope)
    h = bn_relu(h, is_training)
    h = conv(h, 3, 32)
    h = bn_relu(h, is_training)
    h = conv_transp(h, 2, 3)


