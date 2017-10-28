from typing import Tuple, List
import numpy as np
import random
from abc import *

import sys; sys.path.append('.')  # seems to be required to access sibling modules with "data." 
from data.dataset_dir import load_images, load_labels, load_info


class Dataset:
    """
        TODO: rng seed, shuffle list of indices, not images -> add unshuffle function
    """
    def __init__(self, images: list, labels: list, class_count: int):
        self._images = images
        self._labels = labels
        self._class_count = class_count

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._images[key.start:key.stop:key.step], self.labels[key.start:key.stop:key.step]
        else:  # int
            return self._images[key], self.labels[key]

    @property
    def size(self) -> int:
        return len(self)

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self.images[0].shape

    @property
    def class_count(self):
        return self._class_count

    def shuffle(self, order_determining_number: float = -1):
        """ Shuffles the data. """
        image_label_pairs = list(zip(self.images, self.labels))
        if order_determining_number < 0:
            random.shuffle(image_label_pairs)
        else:
            random.shuffle(image_label_pairs, lambda: order_determining_number)
        self.images[:], self.labels[:] = zip(*image_label_pairs)

    def split(self, start, end):
        """ Splits the dataset into two smaller datasets. """
        first = Dataset(self.images[start:end], self.labels[start:end], self.class_count)
        second = Dataset(self.images[:start] + self.images[end:], self.labels[:start] + self.labels[end:],
                         self.class_count)
        return first, second

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def get_an_example(self):
        i = random.randint(0, self.size - 1)
        return self.images[i], self.labels[i]

    @staticmethod
    def load(dataset_directory: str):
        images = load_images(dataset_directory)
        labels = load_labels(dataset_directory)
        class_count = load_info(dataset_directory).class_count
        return Dataset(images, labels, class_count)


class MiniBatchReader:
    def __init__(self, dataset: Dataset, batch_size: int):
        self.current_batch_number = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_batches = dataset.size // batch_size

    def reset(self, shuffle: bool = False):
        if shuffle:
            self.dataset.shuffle()
        self.current_batch_number = 0

    def get_next_batch(self):
        """ Return the next `batch_size` image-label pairs. """
        end = self.current_batch_number + self.batch_size
        if end > self.dataset.size:  # Finished epoch
            return None
        else:
            start = self.current_batch_number
        self.current_batch_number = end
        return self.dataset[start:end]

    def get_generator(self):
        b = self.get_next_batch()
        while b is not None:
            b = self.get_next_batch()
            yield b

