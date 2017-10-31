import cv2  # Can it be replaced with matplotlib and skimage?
import numpy as np


class Visualizer:
    def __init__(self, name='default'):
        self.name = name

    def __del__(self):
        cv2.destroyAllWindows()
        for i in range(4):
            cv2.waitKey(1)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def display_with_labels(self, image, labels):
        """ 
            Displays image mixed with lables alongside with labels only until a 
            key is pressed and returns the key.
            `labels` are WxH images with pixel-values in {0..C}, where W is 
            width, H height, and C the number non-background of classes (0 is 
            background (unlabeled) and should be black)
        """
        labels = cv2.applyColorMap(labels, cv2.COLORMAP_HSV)
        a = np.concatenate(
            (image, cv2.addWeighted(labels, 0.5, image, 0.5, 0),
             (labels * 0.7).astype(np.uint8)),
            axis=1)
        cv2.imshow(self.name, a)
        key = cv2.waitKey(0) & 255
        return key

    def display_image(self, image):
        """ Displays image until a key is pressed and returns the key. """
        cv2.imshow(self.name, image)
        key = cv2.waitKey(0) & 255
        return key

