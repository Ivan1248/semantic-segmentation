import datetime
import numpy as np
import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*


# adds timestamp to `print`
def printt(x):
    print("[{}] {}".format(datetime.datetime.now().strftime('%H:%M:%S'), x))


class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_data(self):
        from data.preparers import Iccv09Preparer
        from data import Dataset, MiniBatchReader
        import shutil

        data_path = os.path.join(
            os.path.dirname(__file__), '../storage/datasets/iccv09')
        prepared_data_path = data_path + '.prepared'
        printt("Deleting prepared dataset if it already exists...")
        if os.path.exists(prepared_data_path):
            shutil.rmtree(
                prepared_data_path, ignore_errors=False, onerror=None)
        printt('Preparing dataset in "{}". This will probably take several minutes...'.format(prepared_data_path))
        assert (prepared_data_path == Iccv09Preparer.prepare(data_path))
        printt('Dateset prepared.')

        printt('Loading dataset...')
        dataset = Dataset.load(prepared_data_path)

        printt('Shuffling dataset...')
        dataset.shuffle()

        printt('Splitting dataset...')
        fold_count = 5
        for f in range(0, fold_count):
            test_part = (np.array([f, f + 1]) / fold_count * dataset.size + 0.5).astype(int)
            test_data, train_data = dataset.split(*test_part)

    def test_data_preparers(self):
        import data.preparers
        import data.preparers.abstract_preparer
        import data.preparers.iccv09_preparer

    def test_data_dataset_dir(self):
        from data.dataset_dir import load_image, save_image
        import matplotlib.pyplot as plt
        plt.ion()
        dirpath = os.path.join(os.path.dirname(__file__), "_test")
        image = load_image(os.path.join(dirpath, "kokosi.png"))
        save_image(image, dirpath, "saved")
        save_path = os.path.join(dirpath, "saved.png")
        image = load_image(save_path)
        import skimage.util
        skimage.io.imshow(
            skimage.util.pad(
                image, ((10, 2), (40, 80), (0, 0)), mode='constant'))
        plt.pause(0.5)
        os.remove(save_path)
        print(os.listdir(dirpath))


if __name__ == "__main__":
    unittest.main()