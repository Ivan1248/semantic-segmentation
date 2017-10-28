import datetime
import os
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
import abc

class AbstractModel(object):
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

    def test(self, dataset: Dataset, evaluator=None):
        if dataset is None:
            dataset = self.test_data
        self._log('Testing...')
        cost_sum, accuracy_sum = 0, 0
        dr = MiniBatchReader(dataset, self._batch_size)
        for _ in range(dr.number_of_batches):
            images, labels = dr.get_next_batch()
            fetches = [self._cost, self._accuracy]
            if evaluator is not None:
                fetches += [self._out_soft]
            foo = self._run_session(fetches, images, labels)
            if evaluator is not None:
                evaluator(foo[2])
            cost_sum += foo[0]
            accuracy_sum += foo[1]
            interrupt = self._check_interrupt(images, labels)
            if interrupt == 'b':
                break
        cost = cost_sum / dr.number_of_batches
        accuracy = accuracy_sum / dr.number_of_batches
        if dataset is self.test_data:
            self.test_accuracy = accuracy
        self._log('cost {:.4f}, accuracy {:.3f}: '.format(cost, accuracy))

    def predict(self, images: list, probs=False):
        """
            Requires pixelwise-class probabilities to be the TensorFlow graph
            node `self_probs`. Feel free to override if needed.
        """
        probs = [
            self._run_session([self._probs], [images[i]])[0]  ## TODO: self._pr
            for i in range(len(images))obs
        ]
        if not probs:
            return [processing.one_hot_to_dense(r) for r in probs]
        return probs

    @abc.abstractmethod
    def _build_graph(self):
        """ 
            Override this. It will be automatically called by the constructor
            (assuming super().__init__(...) is called in the constructor of the
            subclass).
         """
         pass

    def _run_session(self, fetches: list, images, labelings=None):
        feed_dict = {self.input: images}
        if labelings is not None:
            feed_dict[self._y_true] = np.array([
                processing.dense_to_one_hot(labelings[k], self._class_count)
                for k in range(len(labelings))
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
