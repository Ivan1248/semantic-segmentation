import datetime
import os
import re
import postprocessing
import processing
import tensorflow as tf
from semsegnet_core import SemSegNetCore, CallbackResponse
from util import file, console
from util.display_window import DisplayWindow
from data.dataset import Dataset
from data.dataset_reader import MiniBatchReader


class Optimizers:
    gd1em1 = ["gd", tf.train.GradientDescentOptimizer, 1e-1]
    rmsprop2em3 = ["rmsprop", tf.train.RMSPropOptimizer, 2e-3]
    rmsprop1em3 = ["rmsprop", tf.train.RMSPropOptimizer, 1e-3]
    adagrad5em1 = ["adagrad", tf.train.AdagradOptimizer, 5e-1]
    adam = ['adam', tf.train.AdamOptimizer]
    adam2 = ['adam', tf.train.AdamOptimizer, 2e-3]


class SemSegNetB:
    def __init__(self, test_data: Dataset, train_data: Dataset, batch_size: int,
                 layer_count=3, stage_count=3, first_layer_depth=16,
                 activation_func=tf.nn.relu, conv_size=7, optimizer=Optimizers.rmsprop2em3,
                 save_path="storage/models"):
        self.core = SemSegNetCore(
            self.train_data.get_an_example()[0].shape,
            train_data.class_count, layer_count,
            stage_count, first_layer_depth,
            activation_func,
            conv_size,
            optimizer)

        self.log = ""
        self.train_accuracy, self.test_accuracy = 0, 0

        self.train_data, self.test_data = None, None
        self.set_data(test_data, train_data)
        self._batch_size = batch_size
        self.epochs_completed = 0

        self.train_log_period = 5

        self.save_path = save_path

    def __del__(self):
        del self.core

    def __str__(self):
        return "{}-{}stg-{}lay-{}deep,{}".format(self._optimizer_str,
                                                 self.stage_count, 
                                                 self.layer_count,
                                                 self._first_layer_depth,
                                                 self.name)

    def save_state(self, path=None):
        state_string = "{}epochs-{}acc".format(
            self.epochs_completed, int(self.test_accuracy * 100))
        if path is None:
            path = self.save_path
        path = os.path.join(path, str(self))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, state_string)
        self.core.save_state(path)
        file.write_all_text(path + ".log", self.log)
        print("State saved as '" + path + "'.")
        return path

    def load_state(self, path):
        try:
            r = re.search('(-?\d*)epochs-(\d*)acc', path)
            self.epochs_completed = int(r.group(1))
            self.test_accuracy = int(r.group(2)) / 100
        except:
            self.epochs_completed = -10000000
        self.core.load_state(path)
        try:
            self.log = file.read_all_text(path + ".log")
        except:
            self.log = "Log file not found.\n"
        self._log(
            "State loaded (" + str(self.epochs_completed) + " epochs completed).")

    def set_data(self, test_data: Dataset, train_data: Dataset):
        self.test_data, self.train_data = test_data, train_data
        self._log("Dataset loaded (size: " +
                  str(self.train_data.size) + "+" + str(self.test_data.size) + ", shape: " +
                  str(self.core._datapoint_shape) + ", class count: " + str(self.core._class_count) + ").")

    def train(self, epochs=1, keep_prob=1.0):
        mbr = MiniBatchReader(self.train_data, self._batch_size)
        self._log('\nStarting training and evaluation (epochs: {:.2f} ({} batches of size {})))'
                  .format(epochs, epochs * mbr.number_of_batches, self._batch_size))
        abort = False
        alpha = 1 - 2 ** (-0.005 * self.train_log_period * self._batch_size)

        def train_log(self, minibatch, cost, accuracy):
            self._log(' epoch {:d}, step {:d}, cost {:.4f}, accuracy {:.3f} ~{:.3f}'
                      .format(self.epochs_completed + 1, minibatch, cost, accuracy, self.train_accuracy))

        def minibatch_completed(cost, accuracy, b, images, labels):
            nonlocal alpha, abort
            if b % self.train_log_period == 0:
                self.train_accuracy += alpha * (accuracy - self.train_accuracy)
                train_log(b, cost, accuracy)
            interrupt = self._check_interrupt(images, labels)
            if interrupt == 'x':
                abort = True
                return CallbackResponse.abort
            elif interrupt == 'b':
                return CallbackResponse.abort
            return CallbackResponse.proceed

        for ep in range(epochs):
            if abort:
                break
            self._log('\nTraining:')
            mbr.reset(shuffle=True)
            self.core.train(mbr, keep_prob, minibatch_completed)
            self.epochs_completed += 1
            self.test()
            self.test(self.train_data, 'Train')

    def test(self, dataset=None, name='Test', minibatch_out_callback=lambda p: None):
        if dataset is None:
            dataset = self.test_data
        self._log('\nTesting...')

        def minibatch_completed(predictions, images, labels):
            interrupt = self._check_interrupt(images, labels)
            minibatch_out_callback(predictions)
            return CallbackResponse.abort if interrupt == 'x' else CallbackResponse.proceed

        cost, accuracy = self.core.test(MiniBatchReader(
            dataset, self._batch_size), minibatch_completed)
        if dataset is self.test_data:
            self.test_accuracy = accuracy
        self._log(
            name + ': cost {:.4f}, accuracy {:.3f}: '.format(cost, accuracy))

    def run(self, images: list, no_postprocess=False):
        return self.core.run(images, no_postprocess)

    def _log(self, text: str):
        self.log += datetime.datetime.now().strftime('%H:%M:%S') + text + '\n'
        print(text)

    def _check_interrupt(self, images, labels):
        cmd = console.read_line(wait=False, skip_to_last_line=True)
        if cmd is not None:
            if cmd == 't' or cmd == 'test':
                self.test()
            elif cmd == 'b' or cmd == 'break':
                return 'b'
            elif cmd == 'x' or cmd == 'exit':
                return 'x'
            elif cmd == 's' or cmd == 'save':
                self.save_state()
            elif cmd == 'd' or cmd == 'display':
                self._compute_and_display_results(images, labels)
        return 'c'

    def _compute_and_display_results(self, images, labelings):
        all_labelings = [None for _ in range(2 * len(images))]
        i = 0
        w = DisplayWindow('results')
        while True:
            image = images[i // 2]
            if all_labelings[i] is None:
                result, = self.core.run([image])
                # evaluation.compute_class_accuracy(processing.dense_to_one_hot(result, self._class_count, 0),
                # processing.dense_to_one_hot(labelings[i // 2],
                # self._class_count))
                all_labelings[i] = processing.one_hot_to_dense(result)
                all_labelings[i + 1] = labelings[i // 2]
            lab = all_labelings[i]
            if i % 2 == 0:
                lab, = postprocessing.label_segments([image], [lab])
            lab = processing.denormalize(lab * (1 / self.core._class_count))
            key = w.display_with_labels(image, lab)
            if key == 27:
                break
            elif key == ord('b'):
                i -= 2
            i = (i + 1) % (2 * len(images))
        del w


if __name__ == '__main__':
    from scripts.train import train

    data_path = 'storage/datasets/iccv09Data'
    models_path = 'storage/models'
    train(data_path, models_path)

# "GTX 970" 43 puta brža od "Intel® Pentium(R) CPU 2020M @ 2.40GHz × 2"
