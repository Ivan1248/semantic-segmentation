import datetime
import os
import re
import preprocessing
import processing
import postprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from util import file, console
from util.display_window import DisplayWindow
from data.dataset import Dataset
from data.dataset_reader import MiniBatchReader
from util.training_visitor import DummyTrainingVisitor
from abstract_model import AbstractModel


class Baseline1(AbstractModel):
    """ Suggested fixed hyperparameters: 
            activation: ReLU
            optimizer: Adam/RMSProp/SGD
    """

    def __init__(
            self,
            input_shape,  # maybe the network could accept variable image sizes
            class_count,
            batch_size: int,
            save_path="storage/models",
            name='SS-DCNN'):
        super().__init__(self, input_shape, class_count, batch_size, save_path,
                         name)

    def _build_graph(self):
        """ 
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
         """
        from tf_utils import conv_weight_variable, bias_variable, conv2d, max_pool, rescale, pixelwise_softmax

        def layer_depth(layer: int):
            return 3 if layer == -1 else (self._first_layer_depth * 4**layer)

        def layer_shape(int, layer: int) -> tuple:
            div = 2**(layer + (0 if layer == (self.layer_count - 1) else 1))
            return self._shape[0] // div, self._shape[1] // div, \
                layer_depth(layer)

        # Preprocessed input
        sh = layer_shape(0)
        input_shape = [None, sh[0], sh[1], sh[2]]
        self.input = tf.placeholder(tf.float32, shape=input_shape)

        # Convolution layers
        w_conv = np.zeros(self.layer_count, dtype=object)
        b_conv = np.zeros(self.layer_count, dtype=object)
        h_conv = np.zeros(self.layer_count, dtype=object)
        h_pool = np.zeros(self.layer_count, dtype=object)

        w_conv[0] = conv_weight_variable(3, 3, layer_depth(0))
        b_conv[0] = bias_variable(layer_depth(0))
        for l in range(1, self.layer_count):
            w_conv[l] = conv_weight_variable(self._conv_dim,
                                             layer_depth(l - 1),
                                             layer_depth(l))
            b_conv[l] = bias_variable(layer_depth(l))

        split = max(int(5 / 8 * layer_depth(0) + 0.1), layer_depth(0) - 1)

        h1 = conv2d(self.input[:, :, :, :1],
                    w_conv[0][:, :, :1, :split]) + b_conv[0][:split]
        h2 = conv2d(self.input[:, :, :, 1:],
                    w_conv[0][:, :, 1:, split:]) + b_conv[0][split:]
        h_conv[0] = self._act_fun(tf.concat(3, [h1, h2]))
        for l in range(1, self.layer_count):
            h_pool[l - 1] = max_pool(h_conv[l - 1], 2)
            h_conv[l] = self._act_fun(
                conv2d(h_pool[l - 1], w_conv[l]) + b_conv[l])

        # Concatenated feature maps
        fm = h_conv[self.layer_count - 1]

        # 1x1 conolution
        w_fconv = conv_weight_variable(
            1,
            layer_depth(self.layer_count - 1) * self.stage_count,
            self._class_count)
        probs_small = pixelwise_softmax(conv2d(fm, w_fconv))
        probs = rescale(probs_small, 2**(self.layer_count - 1))

        # Training and evaluation
        self._y_true = tf.placeholder(
            tf.float32,
            shape=[None, self._shape[0], self._shape[1], self._class_count])
        self._cost = - \
            tf.reduce_mean(
                self._y_true * tf.log(tf.clip_by_value(probs, 1e-10, 1.0)))
        self._train_step = self._optimizer.minimize(self._cost)
        correct_prediction = tf.equal(
            tf.argmax(probs, 3), tf.argmax(self._y_true, 3))
        self._accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

        self._out_soft = probs[0, :, :, :]

    def train(self,
              train_data: Dataset,
              epoch_count: int = 1,
              visitor=DummyTrainingVisitor()):
        """
            Training consist of epochs. An epoch is a pass through the whole 
            training dataset. An epoch consists of backpropagation steps. A step
            one backpropagation pass (with a mini-batch), ie. one update of all
            parameters.
        """
        dr = MiniBatchReader(train_data, self._batch_size)

        self._log(
            'Starting training and evaluation (epochs: {:.2f} ({} batches of size {} per epoch)))'
            .format(epoch_count, dr.number_of_batches, self._batch_size))
        alpha = 1 - \
            2 ** (-0.005 * train_accuracy_log_period * self._batch_size)
        abort = False

        for ep in range(epoch_count):
            if end:
                break
            self._log('Training:')
            dr.reset(shuffle=True)
            for b in range(dr.number_of_batches):
                images, labels = dr.get_next_batch()
                fetches = [self._train_step, self._cost, self._accuracy]
                _, cost, batch_accuracy = self._run_session(
                    fetches, images, labels)
                if b % train_accuracy_log_period == 0:
                    self.train_accuracy += alpha * \
                        (batch_accuracy - self.train_accuracy)
                    t = datetime.datetime.now().strftime('%H:%M:%S')
                    self._log(
                        t +
                        ' epoch {:d}, step {:d}, cost {:.4f}, accuracy {:.3f} ~{:.3f}'
                        .format(ep, b, cost, batch_accuracy,
                                self.train_accuracy))

                if visitor.minibatch_completed(b, images, labels) == True:
                    end = True
            if visitor.epoch_completed(ep, images, labels) == True:
                end = True


if __name__ == '__main__':
    from scripts.train import train
    data_path = 'storage/datasets/iccv09Data'
    models_path = 'storage/models'
    train(data_path, models_path)

# "GTX 970" 43 times faster than "Intel Pentium 2020M @ 2.40GHz × 2"
