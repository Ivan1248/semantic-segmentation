import abc
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import data
from data import Dataset, MiniBatchReader
from processing.labels import one_hot_to_dense, dense_to_one_hot
from util import file
from util.training_visitor import DummyTrainingVisitor


class AbstractModel(object):
    def __init__(
            self,
            input_shape,  # [width, height, number of channels], maybe [None, None, number of channels] could be allowed too for variable image size
            class_count,
            batch_size: int,  # mini-batch size
            save_path="storage/models",
            name='SS-DCNN'):
        self.name = name

        self._batch_size = batch_size
        self._input_shape, self._class_count = input_shape, class_count

        self._sess = tf.Session()
        self._input, self._y_true, self._probs = self._build_graph()
        self._sess.run(tf.initialize_all_variables())
        
        self.log = []
        self.save_path = save_path
        self._saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=2)

    def __del__(self):  # I am not sure whether this is good
        self._sess.close()
        ops.reset_default_graph()

    def __str__(self):
        return self.name

    def save_state(self, file_path, save_log=True):
        """
            Saves the trained model as `file_path`.
            If `save_log == True`, `self.log` is saved as `file_path`+'.log'.
        """
        file_path = os.path.join(file_path, str(self))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self._saver.save(self._sess, file_path)
        with open(path + ".log", mode='w') as fs:
            fs.write("\n".join(self.log))
            fs.flush()
        print("State saved as '" + file_path + "'.")
        return file_path

    def load_state(self, path):
        self._saver.restore(self._sess, path)
        try:
            self.log = file.read_all_lines(path + ".log")
        except:
            self.log = "Log file not found."
        self._log("State loaded (" + str(self.epochs_completed) +
                  " epochs completed).")

    def predict(self, images: list, probs=False):
        """
            Requires the pixelwise-class probabilities TensorFlow graph node
            to be referenced by `self._probs`.
            It would be good to modify it to do forward propagation in batches
            istead of single images.
        """
        predict_probs = lambda im: self._run_session([self._probs], [im])[0]
        probs = [predict_probs(im) for _, im in enumerate(images)]
        if not probs:
            return [one_hot_to_dense(p) for p in probs]
        return probs

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def test(self, dataset: Dataset, evaluator=None):
        pass

    @abc.abstractmethod
    def _build_graph(self):
        """ 
            Builds the TensorFlow graph for the model.
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
            Returns tuple (input node, target labels node, probs node) (nodes 
            are of type tf.Tensor, the first 2 being placeholders)
         """
        return None, None, None

    def _run_session(self, fetches: list, images, labels=None):
        feed_dict = {self._input: images}
        if labels is not None:
            feed_dict[self._y_true] = np.array([
                dense_to_one_hot(lab, self._class_count)
                for i, lab in enumerate(labels)
            ])
        return self._sess.run(fetches, feed_dict)

    def _log(self, text: str):
        self.log += [text]
        print(text)