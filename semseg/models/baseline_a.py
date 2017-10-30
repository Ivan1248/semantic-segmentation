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
                 name='BaselineA'):
        self.conv_layer_count = 2
        self.learning_rate = 1e-1
        super().__init__(input_shape, class_count, batch_size, save_path, name)

    def _build_graph(self):
        from tf_utils.layers import conv, max_pool, resize

        def layer_width(layer: int):  # number of features per pixel in layer
            return self._input_shape[2] * 4**(layer + 1)

        input_shape = [None] + list(self._input_shape)
        output_shape = input_shape[:3] + [self._class_count]

        # Input image and labels placeholders
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Hidden layers
        h = conv(input, 3, layer_width(0))
        h = tf.nn.relu(h)
        for l in range(1, self.conv_layer_count):
            h = max_pool(h, 2)
            h = conv(h, 3, layer_width(l))
            h = tf.nn.relu(h)

        # Pixelwise softmax classification
        logits = conv(h, 1, self._class_count)
        probs = tf.nn.softmax(logits)
        probs = resize(probs, 2**(self.conv_layer_count - 1))

        # Training and evaluation
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        cost = -tf.reduce_mean(target * tf.log(clipped_probs))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_step = optimizer.minimize(cost)

        self._cost, self._train_step = cost, train_step

        preds, dense_labels = tf.argmax(probs, 3), tf.argmax(target, 3)
        self._accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return input, target, probs

    def train(self,
              train_data: Dataset,
              validation_data: Dataset = None,
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


def main(epoch_count=1):
    from data import Dataset
    from data.preparers import Iccv09Preparer

    data_path = os.path.join(
        os.path.dirname(__file__), '../storage/datasets/iccv09')
    data_path = Iccv09Preparer.prepare(data_path)
    print("Loading data...")
    ds = Dataset.load(data_path)
    print("Splitting dataset...")
    ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
    ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))
    print("Initializing model...")
    model = BaselineA(
        input_shape=ds.image_shape,
        class_count=ds.class_count,
        batch_size=20,
        save_path="../storage/models",
        name='baseline_a-bs8')
    print("Training model...")
    for i in range(epoch_count):
        print("Training epoch {}".format(i))
        model.train(ds_train, epoch_count=1)
        model.test(ds_val)


if __name__ == '__main__':
    main(epoch_count=200)

# "GTX 970" 43 times faster than "Pentium 2020M @ 2.40GHz × 2"
