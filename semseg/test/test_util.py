import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import util

# import processing.transform - currently requires OpenCV

class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_directory(self):
        import directory
  

if __name__ == "__main__":
    unittest.main()