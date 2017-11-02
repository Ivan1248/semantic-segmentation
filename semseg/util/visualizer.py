import numpy as np
import skimage, matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # semseg/*
from data import Dataset


class Visualizer:
    """
        Press "q" to close window. Press anything else to change the displayed
        composite image.
    """

    def __init__(self, name='default'):
        self.name = name

    def display(self, dataset: Dataset, predictor=None):
        from processing.encoding import normalize
        import matplotlib as mpl
        mpl.use('wxAgg')

        cmap = mpl.cm.get_cmap('hsv')

        colors = [np.zeros(3)] + [
            np.array(cmap(i / (dataset.class_count - 2))[:3])
            for i in range(dataset.class_count - 1)
        ]

        def process_labels(lab):
            plab = np.empty(list(lab.shape) + [3])
            for i in range(lab.shape[0]):
                for j in range(lab.shape[1]):
                    try:
                        plab[i, j, :] = colors[lab[i, j]]
                    except:
                        print(lab[i, j], dataset.class_count)
            return plab

        def fuse(im1, im2, a):
            return a * im1 + (1 - a) * im2

        def get_frame(im, lab):
            nim = normalize(im)
            labs = [process_labels(lab)]
            if predictor is not None:
                labs = [process_labels(predictor(im))] + labs
            flabs = [nim] + [fuse(nim, la, 0.5) for la in labs]
            labs = [nim] + [0.5 * l for l in labs]
            fin = np.concatenate(
                [np.concatenate((t, b), axis=1) for t, b in zip(labs, flabs)],
                axis=0)
            return fin

        i = 0

        def on_press(event):
            nonlocal i
            if event.key == 'a':
                i -= 1
            elif event.key == 'q':
                plt.close(event.canvas.figure)
                return
            else:
                i += 1
            i = i % dataset.size
            imgplot.set_data(get_frame(*dataset[i]))
            fig.canvas.draw()

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)
        imgplot = ax.imshow(get_frame(*dataset[0]))
        plt.show()
