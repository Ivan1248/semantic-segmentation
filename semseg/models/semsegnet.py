import datetime
import os
import re
import preprocessing
import processing
import postprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from util import file, console
from util.display_window import DisplayWindow
from data.dataset import Dataset
from data.dataset_reader import MiniBatchReader

"""
    This is from my bachelor project.
"""

class Optimizers:
    gd1em1 = ["gd", tf.train.GradientDescentOptimizer, 1e-1]
    rmsprop2em3 = ["rmsprop", tf.train.RMSPropOptimizer, 2e-3]
    rmsprop1em3 = ["rmsprop", tf.train.RMSPropOptimizer, 1e-3]
    adagrad5em1 = ["adagrad", tf.train.AdagradOptimizer, 5e-1]
    adam = ['adam', tf.train.AdamOptimizer]
    adam2 = ['adam', tf.train.AdamOptimizer, 2e-3]


class SemSegNet:

    def __init__(self, test_data: Dataset, train_data: Dataset, batch_size: int,
                 layer_count: int=3, stage_count: int=3, first_layer_depth: int=16,
                 activation_func=tf.nn.relu, conv_size=7,
                 optimizer=Optimizers.rmsprop2em3, save_path="storage/models", name=''):
        self.log = ""
        self.train_accuracy, self.test_accuracy = 0, 0
        self.name = name

        self.train_data, self.test_data, self._shape, self._class_count = None, None, None, None
        if test_data is not None and train_data is not None:
            self.set_data(test_data, train_data)
        self._batch_size = batch_size
        self.epochs_completed = 0

        self.layer_count, self.stage_count, self._first_layer_depth = layer_count, stage_count, first_layer_depth
        self._conv_dim = conv_size
        self._act_fun = activation_func

        self._pyr = None
        self._y_true = None
        self._sess = tf.Session()
        self._optimizer_str = optimizer[
            0] + "-" + '-'.join(str(p) for p in optimizer[2:])
        self._optimizer = optimizer[1](*optimizer[2:])
        self._build_graph()

        self.save_path = save_path
        self._saver = tf.train.Saver(
            max_to_keep=10, keep_checkpoint_every_n_hours=2)

        self._sess.run(tf.initialize_all_variables())

    def __del__(self):
        self._sess.close()
        ops.reset_default_graph()

    def __str__(self):
        return "{}-{}stg-{}lay-{}deep,{}".format(self._optimizer_str,
                                                 self.stage_count, self.layer_count, self._first_layer_depth, self.name)

    def save_state(self, path=None):
        state_string = "{}epochs-{}acc".format(
            self.epochs_completed, int(self.test_accuracy * 100))
        if path is None:
            path = self.save_path
        path = os.path.join(path, str(self))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, state_string)
        self._saver.save(self._sess, path)
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
        self._saver.restore(self._sess, path)
        try:
            self.log = file.read_all_text(path + ".log")
        except:
            self.log = "Log file not found.\n"
        self._log(
            "State loaded (" + str(self.epochs_completed) + " epochs completed).")

    def set_data(self, test_data: Dataset, train_data: Dataset):
        self.test_data, self.train_data = test_data, train_data
        self._shape = self.train_data.get_an_example()[0].shape
        self._class_count = self.train_data.class_count
        self._log("Dataset loaded (size: " +
                  str(self.train_data.size) + "+" + str(self.test_data.size) + ", shape: " +
                  str(self._shape) + ", class count: " + str(self._class_count) + ").")

    def train(self, epochs: int = None, train_accuracy_log_period=5, keep_prob=1.0, augment_data=True):
        def distort(images, labelings):
            for k in range(len(images)):
                images[k], labelings[k] = preprocessing.random_transform(
                    images[k], labelings[k])

        dr = MiniBatchReader(self.train_data, self._batch_size)

        epochs = 1 if epochs is None else epochs
        batches = epochs * dr.number_of_batches

        self._log('\nStarting training and evaluation (epochs: {:.2f} ({} batches of size {})))'
                  .format(epochs, batches, self._batch_size))
        alpha = 1 - \
            2 ** (-0.005 * train_accuracy_log_period * self._batch_size)
        end = False

        for ep in range(self.epochs_completed, self.epochs_completed + epochs):
            if end:
                break
            self._log('\nTraining:')
            dr.reset(shuffle=True)
            for b in range(dr.number_of_batches):
                images, labels = dr.get_next_batch()
                if augment_data:
                    distort(images, labels)
                _, cost, batch_accuracy = self._run_session(
                    [self._train_step, self._cost, self._accuracy],
                    images,
                    labels,
                    keep_prob)
                if b % train_accuracy_log_period == 0:
                    self.train_accuracy += alpha * \
                        (batch_accuracy - self.train_accuracy)
                    t = datetime.datetime.now().strftime('%H:%M:%S')
                    self._log(
                        t +
                        ' epoch {:d}, step {:d}, cost {:.4f}, accuracy {:.3f} ~{:.3f}'
                        .format(ep, b, cost, batch_accuracy, self.train_accuracy))

                interrupt = self._check_interrupt(images, labels)
                if interrupt == 'b':
                    break
                if interrupt == 'x':
                    end = True
            self.epochs_completed = ep + 1

            self.test()
            self.test(self.train_data, 'Train')

    def test(self, dataset=None, name='Test', out_callback=None):
        if dataset is None:
            dataset = self.test_data
        self._log('\nTesting...')
        cost_sum, accuracy_sum = 0, 0
        dr = MiniBatchReader(dataset, self._batch_size)
        for _ in range(dr.number_of_batches):
            images, labels = dr.get_next_batch()
            fetches = [self._cost, self._accuracy] + \
                ([] if out_callback is None else [self._out_soft])
            foo = self._run_session(fetches, images, labels)
            if out_callback is not None:
                out_callback(foo[2])
            cost_sum += foo[0]
            accuracy_sum += foo[1]
            interrupt = self._check_interrupt(images, labels)
            if interrupt == 'b':
                break
        cost = cost_sum / dr.number_of_batches
        accuracy = accuracy_sum / dr.number_of_batches
        if dataset is self.test_data:
            self.test_accuracy = accuracy
        self._log(
            name + ': cost {:.4f}, accuracy {:.3f}: '.format(cost, accuracy))

    def run(self, images: list, no_postprocess=False):
        plabelings = [self._run_session([self._out_soft], [images[i]])[
            0] for i in range(len(images))]
        labelings = [processing.one_hot_to_dense(r) for r in plabelings]
        return labelings if no_postprocess else postprocessing.label_segments(images, labelings)

    def _build_graph(self):
        from tf_utils import conv_weight_variable, bias_variable, conv2d, max_pool, rescale, pixelwise_softmax

        def layer_depth(layer: int):
            return 3 if layer == -1 else (self._first_layer_depth * 4 ** layer)

        def layer_shape(stage: int, layer: int) -> tuple:
            div = 2 ** (stage + layer + (0 if layer ==
                                         (self.layer_count - 1) else 1))
            return self._shape[0] // div, self._shape[1] // div, layer_depth(layer)

        self._keep_prob = tf.placeholder('float')

        # Preprocessed input
        self._pyr = np.zeros(self.stage_count, dtype=np.ndarray)
        for s in range(self.stage_count):
            sh = layer_shape(s, -1)
            self._pyr[s] = tf.placeholder(
                tf.float32, shape=[None, sh[0], sh[1], sh[2]])

        # Convolution layers
        w_conv = np.zeros(self.layer_count, dtype=object)
        b_conv = np.zeros(self.layer_count, dtype=object)
        h_conv = np.zeros((self.stage_count, self.layer_count), dtype=object)
        h_pool = np.zeros((self.stage_count, self.layer_count), dtype=object)

        w_conv[0] = conv_weight_variable(self._conv_dim, 3, layer_depth(0))
        b_conv[0] = bias_variable(layer_depth(0))
        for l in range(1, self.layer_count):
            w_conv[l] = conv_weight_variable(
                self._conv_dim, layer_depth(l - 1), layer_depth(l))
            b_conv[l] = bias_variable(layer_depth(l))

        split = max(int(5 / 8 * layer_depth(0) + 0.1), layer_depth(0) - 1)
        for s in range(self.stage_count):
            h1 = conv2d(self._pyr[s][:, :, :, :1], w_conv[0][:, :, :1, :split]) + b_conv[0][:split]
            h2 = conv2d(self._pyr[s][:, :, :, 1:], w_conv[0][:, :, 1:, split:]) + b_conv[0][split:]
            h_conv[s, 0] = self._act_fun(tf.concat(3, [h1, h2]))
            for l in range(1, self.layer_count):
                h_pool[s, l - 1] = max_pool(h_conv[s, l - 1], 2)
                h_conv[s, l] = self._act_fun(
                    conv2d(h_pool[s, l - 1], w_conv[l]) + b_conv[l])

        # Concatenated feature maps
        fm = tf.concat(3, [rescale(h_conv[s, self.layer_count - 1], 2 ** s)
                           for s in range(self.stage_count)])
        fm = tf.nn.dropout(fm, self._keep_prob)  # TODO

        # Per-pixel fully-connected layer and network output
        w_fconv = conv_weight_variable(1, layer_depth(
            self.layer_count - 1) * self.stage_count, self._class_count)
        pred_soft_small = pixelwise_softmax(conv2d(fm, w_fconv))
        pred_soft = rescale(pred_soft_small, 2 ** (self.layer_count - 1))

        # Training and evaluation
        self._y_true = tf.placeholder(tf.float32, shape=[None, self._shape[0], self._shape[1], self._class_count])
        self._cost = - \
            tf.reduce_mean(
                self._y_true * tf.log(tf.clip_by_value(pred_soft, 1e-10, 1.0)))
        self._train_step = self._optimizer.minimize(self._cost)
        correct_prediction = tf.equal(
            tf.argmax(pred_soft, 3), tf.argmax(self._y_true, 3))
        self._accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

        self._out_soft = pred_soft[0, :, :, :]

    def _run_session(self, fetches: list, images, labelings=None, keep_prob=1.0):
        def pyramidize(images):
            pyrs = [[] for _ in range(self.stage_count)]
            for k in range(len(images)):
                pyr = preprocessing.normalized_yuv_laplacian_pyramid(
                    images[k], self.stage_count)
                for s in range(self.stage_count):
                    pyrs[s].append(pyr[s])
            return pyrs

        feed_dict = dict()
        if labelings is not None:
            feed_dict[self._y_true] = np.array(
                [processing.dense_to_one_hot(labelings[k], self._class_count) for k in range(len(labelings))])
        images = pyramidize(images)
        for s in range(self.stage_count):
            feed_dict[self._pyr[s]] = np.array(images[s])
        feed_dict[self._keep_prob] = keep_prob
        return self._sess.run(fetches, feed_dict)

    def _log(self, text: str):
        self.log += text + '\n'
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
                result, = self._run_session(
                    [self._out_soft], [image], [labelings[i // 2]])
                # evaluation.compute_class_accuracy(processing.dense_to_one_hot(result, self._class_count, 0),
                # processing.dense_to_one_hot(labelings[i // 2],
                # self._class_count))
                all_labelings[i] = processing.one_hot_to_dense(result)
                all_labelings[i + 1] = labelings[i // 2]
            lab = all_labelings[i]
            if i % 2 == 0:
                lab, = postprocessing.label_segments([image], [lab])
            lab = processing.denormalize(lab * (1 / self._class_count))
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
