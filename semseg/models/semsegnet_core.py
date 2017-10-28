import preprocessing
import processing
import postprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from data.dataset_reader import MiniBatchReader
from optimizer_configurations import Optimizers
from enum import Enum

#### iskoristiti Optimizer.get_name()

class CallbackResponse(Enum):
    proceed = 0
    abort = 1


class SemSegNetCore:
    def __init__(self, datapoint_shape: tuple, class_count: int, layer_count=3, stage_count=3, first_layer_depth=16,
                 activation_func=tf.nn.relu, conv_size=7, optimizer=Optimizers.rmsprop2em3):
        self._datapoint_shape, self._class_count = datapoint_shape, class_count

        self.layer_count, self.stage_count, self.first_layer_depth = layer_count, stage_count, first_layer_depth
        self.conv_dim = conv_size
        self.activation_func = activation_func

        self._pyr = None
        self._y_true = None
        self._sess = tf.Session()
        self._optimizer_str = optimizer[0] + "-" + '-'.join(str(p) for p in optimizer[2:])
        self._optimizer = optimizer[1](*optimizer[2:])

        self._build_graph()

        self._saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

        self._sess.run(tf.initialize_all_variables())

    def __del__(self):
        self._sess.close()
        ops.reset_default_graph()

    def __str__(self):
        return "{}-{}.{}.{}".format(self._optimizer_str,
                                    self.stage_count, self.layer_count, self.first_layer_depth)

    # Public methods

    def save_state(self, file_path: str):
        self._saver.save(self._sess, file_path)

    def load_state(self, file_path: str):
        self._saver.restore(self._sess, file_path)

    def train(self, mbr: MiniBatchReader, keep_prob=1.0,
              minibatch_callback=lambda c, a, b, i, l: CallbackResponse.proceed):
        def distort(images, labelings):
            for k in range(len(images)):
                images[k], labelings[k] = preprocessing.random_transform(images[k], labelings[k])

        for b in range(mbr.number_of_batches):
            images, labels = mbr.get_next_batch()
            distort(images, labels)
            _, cost, batch_accuracy = self._run_session([self._train_step, self._cost, self._accuracy],
                                                        images, labels, keep_prob)
            if minibatch_callback(cost, batch_accuracy, b, images, labels) == CallbackResponse.abort:
                break

    def test(self, mbr: MiniBatchReader, minibatch_callback=lambda p, i, l: CallbackResponse.proceed):
        cost_sum, accuracy_sum = 0, 0
        for _ in range(mbr.number_of_batches):
            images, labels = mbr.get_next_batch()
            fetches = [self._cost, self._accuracy, self._pred_soft]
            fetch_values = self._run_session(fetches, images, labels)
            cost_sum += fetch_values[0]
            accuracy_sum += fetch_values[1]
            if minibatch_callback(fetch_values[2], images, labels) == CallbackResponse.abort:
                break
        cost = cost_sum / mbr.number_of_batches
        accuracy = accuracy_sum / mbr.number_of_batches
        return cost, accuracy

    def run(self, images: list, no_postprocess=False):
        plabelings = [self._run_session([self._pred_soft], [images[i]])[0] for i in range(len(images))]
        labelings = [processing.one_hot_to_dense(r) for r in plabelings]
        return labelings if no_postprocess else postprocessing.label_segments(images, labelings)

    # Private methods

    def _build_graph(self):
        from tf_utils import conv_weight_variable, bias_variable, conv2d, max_pool, rescale, pixelwise_softmax

        def layer_depth(layer: int):
            return 3 if layer == -1 else (self.first_layer_depth * 4 ** layer)

        def layer_shape(stage: int, layer: int) -> tuple:
            div = 2 ** (stage + layer + (0 if layer == (self.layer_count - 1) else 1))
            return self._datapoint_shape[0] // div, self._datapoint_shape[1] // div, layer_depth(layer)

        self._keep_prob = tf.placeholder('float')

        # Preprocessed input
        self._pyr = np.zeros(self.stage_count, dtype=np.ndarray)
        for s in range(self.stage_count):
            sh = layer_shape(s, -1)
            self._pyr[s] = tf.placeholder(tf.float32, shape=[None, sh[0], sh[1], sh[2]])

        # Convolution layers
        w_conv = np.zeros(self.layer_count, dtype=object)
        b_conv = np.zeros(self.layer_count, dtype=object)
        h_conv = np.zeros((self.stage_count, self.layer_count), dtype=object)
        h_pool = np.zeros((self.stage_count, self.layer_count), dtype=object)

        w_conv[0] = conv_weight_variable(self.conv_dim, 3, layer_depth(0))
        b_conv[0] = bias_variable(layer_depth(0))
        for l in range(1, self.layer_count):
            w_conv[l] = conv_weight_variable(self.conv_dim, layer_depth(l - 1), layer_depth(l))
            b_conv[l] = bias_variable(layer_depth(l))

        split = max(int(5 / 8 * layer_depth(0) + 0.1), layer_depth(0) - 1)
        for s in range(self.stage_count):
            h1 = conv2d(self._pyr[s][:, :, :, :1], w_conv[0][:, :, :1, :split]) + b_conv[0][:split]
            h2 = conv2d(self._pyr[s][:, :, :, 1:], w_conv[0][:, :, 1:, split:]) + b_conv[0][split:]
            h_conv[s, 0] = self.activation_func(tf.concat(3, [h1, h2]))
            for l in range(1, self.layer_count):
                h_pool[s, l - 1] = max_pool(h_conv[s, l - 1], 2)
                h_conv[s, l] = self.activation_func(conv2d(h_pool[s, l - 1], w_conv[l]) + b_conv[l])

        # Concatenated feature maps
        fm = tf.concat(3, [rescale(h_conv[s, self.layer_count - 1], 2 ** s) for s in range(self.stage_count)])
        fm = tf.nn.dropout(fm, self._keep_prob)  # TODO

        # Per-pixel fully-connected layer and network output
        w_fconv = conv_weight_variable(1, layer_depth(self.layer_count - 1) * self.stage_count, self._class_count)
        y_pred_soft_small = pixelwise_softmax(conv2d(fm, w_fconv))
        y_pred_soft = rescale(y_pred_soft_small, 2 ** (self.layer_count - 1))

        # Training and evaluation
        self._y_true = tf.placeholder(tf.float32, shape=[None, self._datapoint_shape[0], self._datapoint_shape[1],
                                                         self._class_count])
        self._cost = -tf.reduce_mean(self._y_true * tf.log(tf.clip_by_value(y_pred_soft, 1e-10, 1.0)))
        self._train_step = self._optimizer.minimize(self._cost)
        correct_prediction = tf.equal(tf.argmax(y_pred_soft, 3), tf.argmax(self._y_true, 3))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._pred_soft = y_pred_soft[0, :, :, :]

    def _run_session(self, fetches: list, images, labelings=None, keep_prob=1.0):
        def pyramidize(images):
            pyrs = [[] for _ in range(self.stage_count)]
            for k in range(len(images)):
                pyr = preprocessing.normalized_yuv_laplacian_pyramid(images[k], self.stage_count)
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

# "GTX 970" 43 puta brža od "Intel® Pentium(R) CPU 2020M @ 2.40GHz × 2"
