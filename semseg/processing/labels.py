import numpy as np


def dense_to_one_hot(labels_dense, class_count):
    """ Converts pixel labels from scalars to one-hot vectors. """
    return np.arange(class_count) == (labels_dense[:, :, None])


def one_hot_to_dense(labels_one_hot):
    """ Converts pixel labels from one-hot vectors to scalars. """
    return np.argmax(labels_one_hot, 2)
