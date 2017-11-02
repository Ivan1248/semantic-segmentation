import numpy as np
import tensorflow as tf
from variables import conv_weight_variable, bias_variable


def conv(x, ksize, output_width, bias=True, return_params=False):
    w = conv_weight_variable(ksize, x.shape[3].value, output_width)
    params = [w]
    h = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    if bias:
        b = bias_variable(output_width)
        params.append(b)
        h += b
    return (h, params) if return_params else h


def max_pool(x, divisor):
    return tf.nn.max_pool(x, [1, divisor, divisor, 1],
                          [1, divisor, divisor, 1], 'SAME')


def resize(x, factor):
    shape = (np.array([d.value
                       for d in x.shape[1:3]]) * factor + 0.5).astype(np.int)
    return x if factor == 1 else tf.image.resize_nearest_neighbor(x, shape)


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
