import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import models
from data import Dataset
from data.preparers import Iccv09Preparer


class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_baseline_a(self):
        from models import BaselineA

        data_path = os.path.join(
            os.path.dirname(__file__), '../storage/datasets/iccv09')
        data_path = Iccv09Preparer.prepare(data_path)
        print("Loading data...")
        ds = Dataset.load(data_path)
        print("Splitting dataset...")
        ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
        ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))
        print("Initializing model...")
        model = BaselineA(
            input_shape=ds.image_shape,
            class_count=ds.class_count,
            batch_size=20,
            save_path="../storage/models",
            name='baseline_a-bs8')
        print("Training model...")
        for i in range(200):
            print("Training epoch {}".format(i))
            model.train(ds_train, epoch_count=1)
            model.test(ds_val)


if __name__ == "__main__":
    unittest.main()