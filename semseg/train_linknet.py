import sys
import os
sys.path.append(os.path.dirname(__file__))  # semseg/*
from data import Dataset
from data.preparers import Iccv09Preparer
import dirs

dimargs = sys.argv[1:]

print("Loading and shuffling data...")
data_path = dirs.DATASETS + '/iccv09'
data_path = Iccv09Preparer.prepare(data_path)
ds = Dataset.load(data_path)
ds.shuffle()
print("Splitting dataset...")
ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))

print("Initializing model...")
from models import LinkNet

def get_linket(input_shape, class_count):
    ksizes = [3, 3]
    model = LinkNet(
        input_shape=input_shape,
        class_count=ds_train.class_count,
        class0_unknown=True,
        batch_size=16,
        learning_rate_policy={
            'boundaries': [60, 120, 160],
            'values': [1e-3 * 0.2**i for i in range(4)]
        },
        weight_decay=5e-4,
        training_log_period=10)
    return model


image_shape, class_count = ds_train.image_shape, ds_train.class_count
model = get_linket(image_shape, class_count)

print("Starting training and validation loop...")
from training import train
train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
"""import datetime
model.save_state(dirs.SAVED_MODELS + '/wrn-%d-%d.' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
"""