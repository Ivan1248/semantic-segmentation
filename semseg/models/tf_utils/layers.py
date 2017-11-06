import numpy as np
import tensorflow as tf
from variables import conv_weight_variable, bias_variable


def conv_with_weights(x, w, b=None, stride=1, dilation=1):
    s, d = [stride] * 2, [dilation] * 2
    h = tf.nn.convolution(
        input=x, filter=w, strides=s, dilation_rate=d, padding='SAME')
    if b is not None:
        h += b
    return h


def conv(x,
         ksize,
         output_width,
         stride=1,
         dilation=1,
         bias=True,
         return_params=False):
    w = conv_weight_variable(ksize, x.shape[3].value, output_width)
    params = [w]
    b = None
    if bias:
        b = bias_variable(output_width)
        params.append(b)
    h = conv_with_weights(x, w, b, stride, dilation)
    return (h, params) if return_params else h


def max_pool(x, stride):
    return tf.nn.max_pool(x, [1, stride, stride, 1], [1, stride, stride, 1],
                          'SAME')


def avg_pool(x, stride):
    return tf.nn.avg_pool(x, [1, stride, stride, 1], [1, stride, stride, 1],
                          'SAME')


def _get_rescaled_shape(x, factor):
    return (np.array([d.value
                      for d in x.shape[1:3]]) * factor + 0.5).astype(np.int)


def rescale_nearest_neighbor(x, factor):
    shape = _get_rescaled_shape(x, factor)
    return x if factor == 1 else tf.image.resize_nearest_neighbor(x, shape)


def rescale_bilinear(x, factor):
    shape = _get_rescaled_shape(x, factor)
    return x if factor == 1 else tf.image.resize_bilinear(x, shape)


"""def batch_normalization(self, x, train=True):
    if train:
        mean, variance = tf.nn.moments(x, [0, 1, 2])
        assign_mean = self.mean.assign(mean)
        assign_variance = self.variance.assign(variance)
        with tf.control_dependencies([assign_mean, assign_variance]):
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, self.beta, self.gamma,
                self.epsilon, self.scale_after_norm)
    else:
        mean = self.ewma_trainer.average(self.mean)
        variance = self.ewma_trainer.average(self.variance)
        local_beta = tf.identity(self.beta)
        local_gamma = tf.identity(self.gamma)
        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, local_beta, local_gamma,
            self.epsilon, self.scale_after_norm)"""
