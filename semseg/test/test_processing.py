import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*

import processing
import processing.image_format
import processing.labels
import processing.shape
# import processing.transform - currently requires OpenCV