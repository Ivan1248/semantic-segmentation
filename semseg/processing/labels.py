import numpy as np


def dense_to_one_hot(labels_dense, class_count, leave_out_count=1):
    """
        Converts pixel labels from scalars to one-hot vectors.
        :param leave_out_count: the number of first classes (don't care) to ignore
        If set to n, the first n classes are ignored making the first output indices
        represents class starting from class n. By default, class 0 is "don't care"
        and is ignored with leave_one_out=1.
    """
    a = (np.arange(class_count) == (labels_dense[:, :, None] - leave_out_count)).astype(np.uint8)
    return a


def one_hot_to_dense(labels_one_hot, left_out_count=1):
    """ Converts pixel labels from one-hot vectors to scalars. """
    return np.argmax(labels_one_hot, 2) + left_out_count
