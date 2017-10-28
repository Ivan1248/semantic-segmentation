import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import abc

import sys; sys.path.append(sys.path[0] + "/../../../..")
print(len(sys.path))
for p in sys.path:
    print(p)
from .. import data
from data import Dataset, MiniBatchReader
from processing.labels import one_hot_to_dense, one_hot_to_dense
from util import file
from util.training_visitor import DummyTrainingVisitor


class AbstractModel(object):
    """ 
        Suggested fixed hyperparameters: 
            activation: ReLU
            optimizer: SGD is good enough
    """

    def __init__(
            self,
            input_shape,  # maybe the network could accept variable image sizes
            class_count,
            batch_size: int,
            save_path="storage/models",
            name='SS-DCNN'):
        self.name = name
        self.log = []
        self.save_path = save_path
        self._saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=2)

        self._batch_size = batch_size
        self._input_shape, self._class_count = input_shape, class_count

        self._sess = tf.Session()
        self._build_graph()
        self._sess.run(tf.initialize_all_variables())

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
        import os
        file_path = os.path.join(file_path, str(self))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self._saver.save(self._sess, file_path)
        file.write_all_text(path + ".log", "\n".join(self.log))
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
            Requires pixelwise-class probabilities to be the TensorFlow graph
            node `self_probs`. Feel free to override if needed.
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

    @property
    @abc.abstractmethod
    def _probs(self):
        """
            Returns the pixelwise-class probabilities (pixelwise softmax) graph
            node.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _y_true(self):
        """ Returns the target one-hot labels graph node. """
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_graph(self):
        """ 
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
         """
        pass

    def _run_session(self, fetches: list, images, labels=None):
        feed_dict = {self.input: images}
        if labels is not None:
            feed_dict[self._y_true] = np.array([
                dense_to_one_hot(labels[k], self._class_count)
                for k in range(len(labels))
            ])
        return self._sess.run(fetches, feed_dict)

    def _log(self, text: str):
        self.log += [text]
        print(text)


if __name__ == '__main__':
    from scripts.train import train
    data_path = 'storage/datasets/iccv09Data'
    models_path = 'storage/models'
    train(data_path, models_path)

# "GTX 970" 43 times faster than "Intel Pentium 2020M @ 2.40GHz × 2"
