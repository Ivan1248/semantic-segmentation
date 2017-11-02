import matplotlib.pyplot as plt
import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
from util.visualizer import Visualizer

# import processing.transform - currently requires OpenCV


class Test(unittest.TestCase):
    def test_visualizer(self):
        from data.preparers import Iccv09Preparer
        from data import Dataset
        data_path = os.path.join(
            os.path.dirname(__file__), '../storage/datasets/iccv09')
        data_path = Iccv09Preparer.prepare(data_path)
        ds = Dataset.load(data_path)
        viz = Visualizer("Test")
        viz.display(ds, lambda x: ds[0][1])


if __name__ == "__main__":
    unittest.main()