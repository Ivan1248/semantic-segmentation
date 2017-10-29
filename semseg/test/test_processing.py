import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
import processing

# import processing.transform - currently requires OpenCV

class Test(unittest.TestCase):
    """Unit tests for googlemaps."""

    def test_image_format(self):
        from processing import image_format

    def test_image_format(self):
        from processing import labels

    def test_image_format(self):
        from processing import shape   

    def test_transform(self):
        pass
        # from processing import transform  # currently requires OpenCV
  

if __name__ == "__main__":
    unittest.main()