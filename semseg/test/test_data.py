import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*


class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_data(self):
        import data

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
        plt.pause(2)
        os.remove(save_path)
        print(os.listdir(dirpath))

    def test_data_preparers(self):
        import data.preparers
        import data.preparers.abstract_preparer
        import data.preparers.iccv09_preparer


if __name__ == "__main__":
    unittest.main()