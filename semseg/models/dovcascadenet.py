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


class BaselineAT(AbstractModel):
    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=32,
                 conv_block_count=3,
                 learning_rate=1e-3,
                 training_log_period=1,
                 name='BaselineA'):
        self.conv_block_count = conv_block_count
        self.learning_rate = learning_rate
        self.completed_epoch_count = 0
        self.class0_unknown = class0_unknown
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self):
        from tf_utils.layers import conv, conv_with_weights, max_pool, rescale_bilinear, avg_pool

        stage_count = 3

        def layer_width(layer: int):  # number of channels (features per pixel)
            return min([8 * 4**(layer + 1), 128])

        input_shape = [None] + list(self.input_shape)
        output_shape = input_shape[:3] + [self.class_count]

        # Input image and labels placeholders
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Downsampled input (to improve speed at the cost of accuracy)
        h = [input] + [rescale_bilinear(input, s) for s in [0.5, 0.25]]

        def avg_pool_block():
            for i in range(stage_count):
                h[i] = avg_pool(h[i], 2)

        def conv_block(depth, width):
            for _ in range(depth):
                h[0], (w, b) = conv(h[0], 3, width, return_params=True)
                h[0] = tf.nn.relu(h[0])
                for i in range(1, stage_count):
                    h[i] = conv_with_weights(h[i], w, b)
                    h[i] = tf.nn.relu(h[i])

        def resize_equal_block(size):
            for i in range(stage_count):
                h[i] = tf.image.resize_nearest_neighbor(h[i], size)

        # Hidden layers
        conv_block(1, layer_width(0))
        for l in range(1, self.conv_block_count):
            avg_pool_block()
            conv_block(2, layer_width(l))
        resize_equal_block(h[0].shape[1:3])
        h = tf.concat(h, axis=3)
        h = conv(h, 3, h.shape[3].value*2//3, dilation=1)
        #h = h[0] * tf.Variable(1.0) + h[1] * tf.Variable(
        #   1.0) + h[1] * tf.Variable(1.0)

        # Pixelwise softmax classification and label upscaling
        logits = conv(h, 1, self.class_count)
        probs = tf.nn.softmax(logits)
        probs = tf.image.resize_bilinear(probs, output_shape[1:3])

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        ts = lambda x: x[:, :, :, 1:] if self.class0_unknown else x
        cost = -tf.reduce_mean(ts(target) * tf.log(ts(clipped_probs)))

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_step = optimizer.minimize(cost)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 3), tf.argmax(target, 3)

        # Other evaluation measures
        self._n_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=cost,
            training_step=training_step)

    def train(self,
              train_data: Dataset,
              validation_data: Dataset = None,
              epoch_count: int = 1):
        self._train(train_data, validation_data, epoch_count, {
            'accuracy': self._n_accuracy
        })

    def test(self, dataset):
        """ Override if extra fetches (maybe some evaluation measures) are needed """
        self._test(dataset, extra_fetches={'accuracy': self._n_accuracy})


def main(epoch_count=1):
    from data import Dataset
    from data.preparers import Iccv09Preparer
    from util import console, visualization
    from util.visualization import Visualizer

    data_path = os.path.join(
        os.path.dirname(__file__), '../storage/datasets/iccv09')
    data_path = Iccv09Preparer.prepare(data_path)
    print("Loading and deterministically shuffling data...")
    ds = Dataset.load(data_path)
    ds.shuffle(order_determining_number=0.5)
    print("Splitting dataset...")
    ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
    ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))
    print("Initializing model...")
    model = BaselineAT(
        input_shape=ds.image_shape,
        class_count=ds.class_count,
        class0_unknown=True,
        batch_size=16,  # 32
        learning_rate=5e-5,
        name='BaselineA-bs16',
        training_log_period=5)


    def handle_epoch(epoch):
        indexes = [1,5,7,8,12,13,21,52,55,56]
        images = ds_val_copy[indexes].images
        plabels = np.array(model.predict(images))[:,14:-15,10:-10]
        frame = visualization.get_multi_result_visualization(
            images[:,14:-15,10:-10], plabels, plabels, ds.class_count, row_count=2)
        from skimage.io import imsave
        imsave("./storage/frames/{}.bmp".format(epoch), frame)

    def handle_step(step):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'd':
            Visualizer().display(ds_val_copy, lambda im: model.predict([im])[0])
        elif text == 'q':
            return True
        return False

    model.training_step_event_handler = handle_step

    print("Starting training and validation loop...")
    #model.test(ds_val)
    ds_val = ds_val[:8*8]
    ds_val_copy = ds_val[:]
    ds_train_copy = ds_train[:]
    handle_epoch(0)
    for i in range(epoch_count):
        model.train(ds_val, epoch_count=1)
        model.test(ds_val_copy)
        handle_epoch(i+1)
    model.save_state()


if __name__ == '__main__':
    main(epoch_count=800)

# "GTX 970" 43 times faster than "Pentium 2020M @ 2.40GHz × 2"

