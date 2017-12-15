import numpy as np
import matplotlib.pyplot as plt
import skimage

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # /*
import matplotlib as mpl


def get_colors(n, black_first=True):
    cmap = mpl.cm.get_cmap('hsv')
    colors = [np.zeros(3)] if black_first else []
    colors += [
        np.array(cmap(0.8 * i / (n - len(colors) - 1))[:3])
        for i in range(n - len(colors))
    ]
    return colors

def fuse_images(im1, im2, a):
    return a * im1 + (1 - a) * im2

def colorify_label(lab, colors):
    plab = np.empty(list(lab.shape) + [3])
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            plab[i, j, :] = colors[lab[i, j]]
    return plab


def compose(images, format='0,0;1,0-1'):
    def get_image(frc):
        inds = [int(i) for i in frc.split('-')]
        assert (len(inds) <= 2)
        ims = [images[i] for i in inds]
        return ims[0] if len(ims) == 1 else fuse_images(ims[0], ims[1], 0.5)

    format = format.split(';')
    format = [f.split(',') for f in format]
    return np.concatenate([
        np.concatenate([get_image(frc) for frc in frow], 1) for frow in format
    ], 0)


def get_result_visualization(image, tlabel, plabel):
    return compose([image, tlabel, plabel], format='0,0;2,0-2;1,0-1')


def get_multi_result_visualization(images,
                                   tlabels,
                                   plabels,
                                   class_count,
                                   row_count=1,
                                   element_format='0;0-2'):
    assert (len(images) == len(tlabels) and len(tlabels) == len(plabels))
    assert (len(images) % row_count == 0)

    colors = get_colors(class_count, black_first=True)

    def get_composition(i):
        nim = normalize(images[i])
        cplab = colorify_label(plabels[i], colors)
        return compose([nim, None, cplab], format=element_format)

    col_count = len(images) // row_count
    return np.concatenate([
        np.concatenate(
            [get_composition(i * col_count + j) for j in range(col_count)], 1)
        for i in range(row_count)
    ], 0)


class Viewer:
    """
        Press "q" to close the window. Press anything else to change the displayed
        composite image. Press "a" to return to the previous image.
    """

    def __init__(self, name=None):
        self.name = name

    def display(self, data, mapping=lambda x: x):
        import matplotlib as mpl
        mpl.use('wxAgg')

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
            i = i % data.size
            imgplot.set_data(mapping(data[i]))
            fig.canvas.set_window_title(str(i))
            fig.canvas.draw()

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)
        fig.canvas.set_window_title('0')
        imgplot = ax.imshow(mapping(data[0]))
        plt.show()


from data import Dataset
#from processing.encoding import normalize

class SemSegViewer:
    """
        Press "q" to close the window. Press anything else to change the displayed
        composite image. Press "a" to return to the previous image.
    """

    def __init__(self, name='default'):
        self.name = name

    def display(self, dataset: Dataset, predictor=None):
        import matplotlib as mpl
        mpl.use('wxAgg')

        colors = get_colors(dataset.class_count, black_first=True)

        def get_frame(im, lab):
            nim = skimage.img_as_float(im)
            clab = colorify_label(lab, colors)
            cplab = colorify_label(predictor(im), colors)
            return get_result_visualization(nim, clab, cplab)

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
            fig.canvas.set_window_title(str(i))
            fig.canvas.draw()

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)
        fig.canvas.set_window_title('0')
        imgplot = ax.imshow(get_frame(*dataset[0]))
        plt.show()
