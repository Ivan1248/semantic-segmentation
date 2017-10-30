import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import models
from data import Dataset
from data.preparers import Iccv09Preparer


class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_baseline_a(self):
        from models import baseline_a
        baseline_a.main()


if __name__ == "__main__":
    unittest.main()