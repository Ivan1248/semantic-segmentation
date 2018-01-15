import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from layers import *

# from semseg.models.tf_utils.layers import conv, conv_transp, residual_block, ResidualBlockKind, bn_relu, max_pool


def linknet_encoder(x,
                    is_training,
                    keep_shape,
                    reuse: bool = None,
                    scope: str = None):
    width = (x.shape[-1].value * (1 if keep_shape else 2))

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


def linknet_decoder(x, is_training = None, width = 32, reuse: bool = None, scope: str = None):
    H, W, C = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    m = int(x.shape[-1])
    m_div4 = m//4
    with tf.variable_scope(scope, 'LinkNetDecoder', reuse=reuse):
        h = conv(
            x, ksize=1, width=m_div4, scope="CONV-1", reuse=reuse)
        h = bn_relu(h, is_training, scope="BN_RELU-1", reuse=reuse)
        h = conv_transp(h, ksize=3, stride=2, width=m_div4, scope="CONV_TRANSP-1", reuse=reuse)
        h = bn_relu(h, is_training, scope="BN_RELU-2", reuse=reuse)
        h = conv(h, ksize=1, width=width, scope="CONV-2", reuse=reuse)
        h = bn_relu(h, is_training, scope="BN_RELU-3", reuse=reuse)
    return h


def linknet(x,
           n_classes,
           is_training,
           reuse: bool = None,
           scope: str = None):

    # encoder_widths = [64, 128, 256, 512]
    encoder_keep_shapes = [True, False, False, False]
    decoder_widths = [64, 64, 128, 256]
    encoder_outputs = []
    with tf.variable_scope(scope, 'LinkNet', reuse=reuse):
        h = conv(x, ksize=7, width=64, stride=2, reuse=reuse, scope="CONV-Init")
        h = bn_relu(h, is_training, reuse=reuse, scope="BN_RELU-Init")
        h = max_pool(h, stride=2, ksize=3)
        print(h.shape)
        print("ENCODER_OUTPUT")
        for i in range(4):
            h = linknet_encoder(h, is_training, encoder_keep_shapes[i], reuse=reuse, scope="Encoder"+str(i+1))
            encoder_outputs.append(h)
            print(h.shape)

        print("DECODER_OUTPUT")
        for i in range(1, 4):
            print(decoder_widths[-i])
            h = linknet_decoder(h, is_training, decoder_widths[-i], reuse=reuse, scope="Decoder"+str(i))
            print(h.shape)
            h = h + encoder_outputs[-(i+1)]

        h = linknet_decoder(h, is_training, decoder_widths[-4], reuse=reuse, scope="Decoder4")
        print(h.shape)
        h = conv_transp(h, ksize=3, width=32, stride=2, reuse=reuse, scope="CONV_TRANSP-Final1")
        h = bn_relu(h, is_training, reuse=reuse, scope="BN_RELU-Final1")
        h = conv(h, 3, 32, reuse=reuse, scope="CONV-Final1")
        h = bn_relu(h, is_training, reuse=reuse, scope="BN_RELU-Final2")
        h = conv_transp(h, ksize=2, width=n_classes, stride=2, reuse=reuse, scope="CONV_TRANSP-Final2")
        # h = bn_relu(h, is_training, reuse=reuse, scope="BN_RELU-Final3")
        return h


