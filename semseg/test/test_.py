import matplotlib.pyplot as plt
import unittest

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
from util.visualizer import Visualizer

# import processing.transform - currently requires OpenCV


class Test(unittest.TestCase):
    def test(self):
        import matplotlib as mpl 
        mpl.use('wxAgg')

        import numpy as np 
        import matplotlib.pyplot as plt

        # just some random data 
        frames = [np.random.random((10, 10)) for _ in range(100)]
        keeped_frames = [] 
        i = 0

        # event listener 
        def press(event):
            nonlocal i
            if event.key == '1':
                print('Appending frame')
                keeped_frames.append(frames[i % 100])
            i += 1
            imgplot.set_data(frames[i % 100])
            fig.canvas.draw()

        fig, ax = plt.subplots() 
        fig.canvas.mpl_connect('key_press_event', press)
        imgplot = ax.imshow(frames[i % 100]) 
        plt.show()


if __name__ == "__main__":
    unittest.main()