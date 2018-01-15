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
from tf_utils import layers, layers_linknet


class LinkNet(AbstractModel):

    def __init__(self,
                 input_shape,
                 class_count,
                 class0_unknown=False,
                 batch_size=32,
                 learning_rate_policy=1e-2,
                 weight_decay=1e-4,
                 training_log_period=1,
                 name='ResNet'):
        self.class_count = class_count
        self.completed_epoch_count = 0
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
        from layers_linknet import linknet

        # Input image and labels placeholders
        input_shape = [None] + list(self.input_shape)
        output_shape = input_shape[:3] + [self.class_count]
        input = tf.placeholder(tf.float32, shape=input_shape)
        target = tf.placeholder(tf.float32, shape=output_shape)

        new_input_shape = [input.shape[0].value] + [256, 320, 3]


        new_input = tf.image.resize_image_with_crop_or_pad(input, target_height=new_input_shape[1], target_width=new_input_shape[2])
        # Hidden layers
        logits = linknet(
            new_input,
            n_classes=self.class_count,
            is_training=is_training)
        new_logits = tf.image.resize_image_with_crop_or_pad(logits, target_height=output_shape[1], target_width=output_shape[2])

        # Pixelwise softmax classification and label upscaling
        probs = tf.nn.softmax(new_logits)
        # probs = tf.image.resize_bilinear(probs, output_shape[1:3])

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
