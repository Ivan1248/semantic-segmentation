import tensorflow as tf

from ioutils import path, console
import processing.preprocessing as pp
from data import Dataset
from models import AbstractModel
import dirs

from visualization import SemSegViewer


def train(model: AbstractModel,
          ds_train: Dataset,
          ds_val: Dataset,
          epoch_count=200):

    def handle_step(step):
        text = console.read_line(impatient=True, discard_non_last=True)
        if text == 'd':
            SemSegViewer().display(ds_val, lambda im: model.predict([im])[0])
        elif text == 'q':
            return True
        return False

    model.training_step_event_handler = handle_step

    from processing.data_augmentation import augment_cifar

    model.test(ds_val)
    ds_train_part = ds_train[:ds_val.size]
    for i in range(epoch_count):
        model.train(ds_train, epoch_count=1)
        model.test(ds_val, 'validation data')
        model.test(ds_train_part, 'training data subset')
