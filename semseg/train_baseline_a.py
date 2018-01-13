import sys
import os
sys.path.append(os.path.dirname(__file__))  # semseg/*
from data import Dataset
from data.preparers import Iccv09Preparer
import dirs

print("Loading and deterministically shuffling data...")
data_path = dirs.DATASETS + '/iccv09'
data_path = Iccv09Preparer.prepare(data_path)
ds = Dataset.load(data_path)
ds.shuffle()
print("Splitting dataset...")
ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))

print("Initializing model...")
from models import BaselineA
model = BaselineA(
    input_shape=ds_train.image_shape,
    class_count=ds_train.class_count,
    class0_unknown=True,
    batch_size=16,
    learning_rate_policy={
        'boundaries': [60, 120, 160],
        'values': [1e-4 * 0.2**i for i in range(4)]
    },
    name='BaselineA-bs16',
    training_log_period=10)

print("Starting training and validation loop...")
from training import train
train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
"""import datetime
model.save_state(dirs.SAVED_MODELS + '/bla' +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
"""