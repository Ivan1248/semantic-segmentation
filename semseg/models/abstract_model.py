import abc
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import data
from data import Dataset, MiniBatchReader
from processing.labels import one_hot_to_dense, dense_to_one_hot
from ioutils import file


class AbstractModel(object):
    class EssentialNodes:
        def __init__(self, input, target, probs, loss, training_step,
                     evaluation: dict):
            self.input = input
            self.target = target
            self.probs = probs
            self.loss = loss
            self.training_step = training_step
            self.evaluation = evaluation

    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable image size
            class_count,
            batch_size: int,  # mini-batch size
            learning_rate_policy,
            training_log_period=1,
            name="DCNN"):
        self.name = name

        self.batch_size = batch_size
        self.learning_rate_policy = learning_rate_policy
        self.input_shape, self.class_count = input_shape, class_count

        self._sess = tf.Session()

        self._epoch = tf.Variable(0, False, dtype=tf.int32, name='epoch')
        self._increment_epoch = tf.assign(
            self._epoch, self._epoch + 1, name='increment_epoch')
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        lr = self.learning_rate_policy
        if not type(lr) is float:
            lr = tf.train.piecewise_constant(
                self._epoch, boundaries=lr['boundaries'], values=lr['values'])
        self.nodes = self._build_graph(lr, self._epoch, self._is_training)

        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

        self.training_log_period = training_log_period
        self.log = []
        self._saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=2)

        self.training_step_event_handler = lambda step: False

    def __del__(self):  # I am not sure whether this is good
        self._sess.close()
        ops.reset_default_graph()

    def __str__(self):
        return self.name

    def save_state(self, path, save_log=True):
        """
            Saves the trained model as `file_path`.
            If `save_log == True`, `self.log` is saved as `file_path`+'.log'.
        """
        file_path = os.path.join(path, str(self))
        if not os.path.exists(path):
            os.makedirs(path)
        self._saver.save(self._sess, file_path)
        with open(file_path + ".log", mode='w') as fs:
            fs.write("\n".join(self.log))
            fs.flush()
        print("State saved as '" + file_path + "'.")
        return file_path

    def load_state(self, path):
        self._saver.restore(self._sess, path)
        self.epoch = self._sess.run(self._epoch)
        try:
            self.log = file.read_all_lines(path + ".log")
        except:
            self.log = "Log file not found."
        self._log("State loaded (" + str(self.epoch) + " epochs completed).")

    def predict(self, images: list, probs=False):
        """
            Requires the pixelwise-class probabilities TensorFlow graph node
            to be referenced by `self.nodes.probs`.
            It would be good to modify it to do forward propagation in batches
            istead of single images.
        """
        pr_probs = self._run([self.nodes.probs], images, None, False)[0]
        return pr_probs if probs else [one_hot_to_dense(p) for p in pr_probs]

    def train(self,
              train_data: Dataset,
              validation_data: Dataset = None,
              epoch_count: int = 1):
        """ Override if extra fetches need to be different """
        self._train(train_data, validation_data, epoch_count,
                    self.nodes.evaluation)

    def test(self, dataset, test_name=None):
        """ Override if extra fetches need to be different """
        self._test(dataset, self.nodes.evaluation, test_name)

    @property
    def epoch(self):
        return self._sess.run(self._epoch)

    def _train_minibatch(self, images, labels, extra_fetches: list = []):
        fetches = [self.nodes.training_step, self.nodes.loss
                   ] + list(extra_fetches)
        evals = self._run(fetches, images, labels, is_training=True)
        cost, extra = evals[1], evals[2:]
        return cost, extra

    def _test_minibatch(self, images, labels, extra_fetches: list = []):
        fetches = [self.nodes.loss] + list(extra_fetches)
        evals = self._run(fetches, images, labels, is_training=False)
        cost, extra = evals[0], evals[1:]
        return cost, extra

    def _train(self,
               train_data: Dataset,
               validation_data: Dataset = None,
               epoch_count: int = 1,
               extra_fetches: dict = dict()):
        def log_training_start(epoch_count, batch_count, batch_size):
            self._log('Training (epochs: {}; {} batches of size {} per epoch)'
                      .format(epoch_count, batch_count, batch_size))

        def handle_step_completed(batch, cost, extra):
            evalstr = lambda k, v: "{} {:5.3f}, ".format(k, v)
            if b % self.training_log_period == 0 or b == dr.number_of_batches - 1:
                eval = evalstr("cost", cost)
                for k, v in zip(extra_fetches.keys(), extra):
                    eval += evalstr(k, v)
                self._log('  {:3d}.{:3d}: {}'.format(self.epoch, b, eval))
            if self.training_step_event_handler(b): end = True

        dr = MiniBatchReader(train_data, self.batch_size)
        log_training_start(epoch_count, dr.number_of_batches, self.batch_size)
        end = False
        for ep in range(epoch_count):
            self._log('epoch {:d}'.format(self.epoch))
            dr.reset(shuffle=True)
            for b in range(dr.number_of_batches):
                images, labels = dr.get_next_batch()
                ef = extra_fetches.values()
                cost, extra = self._train_minibatch(images, labels, ef)
                handle_step_completed(b, cost, extra)
            self._sess.run(self._increment_epoch)
            if end: break

    def _test(self, dataset, extra_fetches: dict = dict(), test_name=None):
        self._log('Testing%s...' %
                  ("" if test_name is None else " (" + test_name + ")"))
        cost_sum, extra_sum = 0, np.zeros(len(extra_fetches))
        dr = MiniBatchReader(dataset, self.batch_size)
        for _ in range(dr.number_of_batches):
            images, labels = dr.get_next_batch()
            cost, extra = self._test_minibatch(images, labels,
                                               extra_fetches.values())
            cost_sum += cost
            extra_sum += np.array(extra)
        cost = cost_sum / dr.number_of_batches
        extra = extra_sum / dr.number_of_batches
        ev = dict(zip(extra_fetches.keys(), extra))
        self._log('  cost {:.4f}, {}'.format(cost, ev))

    @abc.abstractmethod
    def _build_graph(self, learning_rate, epoch, is_training):
        """ 
            Builds the TensorFlow graph for the model.
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
            Returns tuple (input node, target labels node, probs node) (nodes 
            are of type tf.Tensor, the first 2 being placeholders)
         """
        return AbstractModel.EssentialNodes(None, None, None, None, None)

    def _run(self, fetches: list, images, labels=None, is_training=None):
        feed_dict = {self.nodes.input: images}
        if labels is not None:
            feed_dict[self.nodes.target] = np.array([
                dense_to_one_hot(lab, self.class_count)
                for i, lab in enumerate(labels)
            ])
        if self._is_training is not None:
            feed_dict[self._is_training] = is_training
        return self._sess.run(fetches, feed_dict)

    def _log(self, text: str):
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        text = "[{}] {}".format(timestr, text)
        self.log.append(text)
        print(text)

    def _print_vars(self):
        vars = sorted([v.name for v in tf.global_variables()])
        for i, v in enumerate(vars):
            print(i, v)
        exit()
