import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
sys.path.append(os.path.dirname(__file__))  # /models/
from data import Dataset, MiniBatchReader
from ioutils import path

from abstract_model import AbstractModel
from tf_utils import layers


class ResNet(AbstractModel):

    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=32,
                 learning_rate_policy=1e-2,
                 block_kind=layers.ResidualBlockKind([3, 3]),
                 group_lengths=[3, 3, 3],
                 base_width=16,
                 widening_factor=1,
                 weight_decay=1e-4,
                 training_log_period=1,
                 name='ResNet'):
        self.completed_epoch_count = 0
        self.block_kind = block_kind
        self.group_lengths = group_lengths
        self.depth = 1 + sum(group_lengths) * len(block_kind.ksizes) + 1
        self.zagoruyko_depth = self.depth - 1 + len(group_lengths)
        self.base_width = base_width
        self.widening_factor = widening_factor
        self.weight_decay = weight_decay
        self.class0_unknown = class0_unknown
        super().__init__(
            input_shape=input_shape,
            class_count=class_count,
            batch_size=batch_size,
            learning_rate_policy=learning_rate_policy,
            training_log_period=training_log_period,
            name=name)

    def _build_graph(self, learning_rate, epoch, is_training):
        from layers import conv, resnet

        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        output_shape = input_shape[:3] + [self.class_count]
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        # Hidden layers
        h = resnet(
            input,
            base_width=self.base_width,
            widening_factor=self.widening_factor,
            group_lengths=self.group_lengths,
            is_training=is_training)

        # Pixelwise softmax classification and label upscaling
        logits = conv(h, 1, self.class_count)
        probs = tf.nn.softmax(logits)
        probs = tf.image.resize_bilinear(probs, output_shape[1:3])

        # Loss
        clipped_probs = tf.clip_by_value(probs, 1e-10, 1.0)
        ts = lambda x: x[:, :, :, 1:] if self.class0_unknown else x
        loss = -tf.reduce_mean(ts(target) * tf.log(ts(clipped_probs)))

        # Regularization
        vars = tf.global_variables()
        weight_vars = filter(lambda x: 'weights' in x.name, vars)
        l2reg = tf.reduce_sum(list(map(tf.nn.l2_loss, weight_vars)))
        loss += self.weight_decay * l2reg

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)

        # Dense predictions and labels
        preds, dense_labels = tf.argmax(probs, 3), tf.argmax(target, 3)

        # Other evaluation measures
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, dense_labels), tf.float32))

        return AbstractModel.EssentialNodes(
            input=input,
            target=target,
            probs=probs,
            loss=loss,
            training_step=training_step,
            evaluation={'accuracy': accuracy})
