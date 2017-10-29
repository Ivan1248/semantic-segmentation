import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
from data import Dataset, MiniBatchReader
from util.training_visitor import DummyTrainingVisitor

from abstract_model import AbstractModel


class BaselineA(AbstractModel):
    def __init__(self,
                 input_shape,
                 class_count,
                 batch_size: int,
                 save_path="storage/models",
                 name='SS-DCNN'):
        self.layer_count = 3
        super().__init__(input_shape, class_count, batch_size, save_path, name)

    def _build_graph(self):
        from tf_utils import conv_weight_variable, bias_variable, conv2d, max_pool, rescale, pixelwise_softmax

        def layer_width(layer: int):
            return 3 if layer == -1 else (self._input_shape[2] * 4**layer)

        # Input image and labels placeholders
        input_shape = [None] + list(self._input_shape)
        output_shape = input_shape[:3] + [self._class_count]
        image = tf.placeholder(tf.float32, shape=input_shape)
        target_labels = tf.placeholder(tf.float32, shape=output_shape)

        # Hidden layers
        w_conv = np.zeros(self.layer_count, dtype=object)
        b_conv = np.zeros(self.layer_count, dtype=object)
        h_conv = np.zeros(self.layer_count, dtype=object)
        h_pool = np.zeros(self.layer_count, dtype=object)

        w_conv[0] = conv_weight_variable(3, 3, layer_width(0))
        b_conv[0] = bias_variable(layer_width(0))
        for l in range(1, self.layer_count):
            w_conv[l] = conv_weight_variable(3,
                                             layer_width(l - 1),
                                             layer_width(l))
            b_conv[l] = bias_variable(layer_width(l))

        h_conv[0] = tf.nn.relu(conv2d(image, w_conv[0]) + b_conv[0])
        for l in range(1, self.layer_count):
            h_pool[l - 1] = max_pool(h_conv[l - 1], 2)
            h_conv[l] = tf.nn.relu(
                conv2d(h_pool[l - 1], w_conv[l]) + b_conv[l])

        # Pixelwise softmax classification
        w_fconv = conv_weight_variable(
            1,
            layer_width(self.layer_count - 1),
            self._class_count)
        probs_small = pixelwise_softmax(conv2d(h_conv[-1], w_fconv))
        probs = rescale(probs_small, 2**(self.layer_count - 1))

        # Training and evaluation
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        self._cost = -tf.reduce_mean(target_labels * tf.log(clipped_probs))
        self._train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(self._cost)

        preds, dense_labels = tf.argmax(clipped_probs, 3), tf.argmax(target_labels, 3)
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return image, target_labels, probs

    def train(self,
              train_data: Dataset,
              epoch_count: int = 1,
              visitor=DummyTrainingVisitor()):
        dr = MiniBatchReader(train_data, self._batch_size)
        train_accuracy_log_period = 1

        self._log(
            'Starting training and evaluation (epochs: {} ({} batches of size {} per epoch)))'
            .format(epoch_count, dr.number_of_batches, self._batch_size))
        end = False

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
                    t = datetime.datetime.now().strftime('%H:%M:%S')
                    self._log(
                        t +
                        ' epoch {:d}, step {:d}, cost {:.4f}, accuracy {:.3f}'
                        .format(ep, b, cost, batch_accuracy))
                #if visitor.minibatch_completed(b, images, labels) == True:
                 #   end = True
            #if visitor.epoch_completed(ep, images, labels) == True:
            #    end = True


if __name__ == '__main__':
    #from scripts.train import train
    data_path = 'storage/datasets/iccv09Data'
    models_path = 'storage/models'

    #train(data_path, models_path)

# "GTX 970" 43 times faster than "Intel Pentium 2020M @ 2.40GHz × 2"
