import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*

class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_data(self):
        import data
    
    def test_data_preparers(self):
        import data.preparers
        import data.preparers.abstract_preparer
        import data.preparers.iccv09_preparer

if __name__ == "__main__":
    unittest.main()