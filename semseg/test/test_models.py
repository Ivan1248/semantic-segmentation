import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import models

class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_abstract_model(self):
        from models.abstract_model import AbstractModel
        

if __name__ == "__main__":
    unittest.main()