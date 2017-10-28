import tensorflow as tf


def conv_weight_variable(size: int, in_channels: int, out_channels: int):
    shape = [size, size, in_channels, out_channels]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, mean=0))


def bias_variable(n: int):
    return tf.Variable(tf.constant(0.1, shape=[n]))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, dim: int):
    return tf.nn.max_pool(x, [1, dim, dim, 1], [1, dim, dim, 1], 'SAME')


def rescale(x, factor):
    shape = x.get_shape()[1:3]*factor
    return x if factor == 1 else tf.image.resize_nearest_neighbor(x, shape)


def pixelwise_softmax(input: tf.Tensor):
    shape = input.get_shape()
    s = [-1] + [int(shape[i]) for i in range(1, 4)]
    return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, 9])), s)


"""def normalize(self, x, train=True):
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
