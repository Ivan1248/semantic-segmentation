import sys
import os
from data import Dataset
from data.preparers import Iccv09Preparer
import dirs

dimargs = sys.argv[1:]
if len(dimargs) not in [0, 2]:
    print("usage: train-wrn.py [<Zagoruyko-depth> <widening-factor>]")
zaggydepth, k = (16, 4) if len(dimargs) == 0 else map(int, dimargs)

print("Loading and shuffling data...")
data_path = dirs.DATASETS + '/iccv09'
data_path = Iccv09Preparer.prepare(data_path)
ds = Dataset.load(data_path)
ds.shuffle()
print("Splitting dataset...")
ds_trainval, ds_test = ds.split(0, int(ds.size * 0.8))
ds_train, ds_val = ds_trainval.split(0, int(ds_trainval.size * 0.8))

print("Initializing model...")
from models import ResNet
from models.tf_utils.layers import ResidualBlockKind


def get_wide_resnet(n, k, input_shape, class_count, dim_increase='conv1'):
    group_count = 3
    ksizes = [3, 3]
    blocks_per_group = (n - 4) // (group_count * len(ksizes))
    print("group count: {}, blocks per group: {}".format(
        group_count, blocks_per_group))
    model = ResNet(
        input_shape=input_shape,
        class_count=ds_train.class_count,
        class0_unknown=True,
        batch_size=16,
        learning_rate_policy={
            'boundaries': [60, 120, 160],
            'values': [1e-1 * 0.2**i for i in range(4)]
        },
        block_kind=ResidualBlockKind(
            ksizes=ksizes,
            dropout_locations=[0],
            dropout_rate=0.3,
            dim_increase=dim_increase),
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=k,
        weight_decay=5e-4,
        training_log_period=10)
    assert n == model.zagoruyko_depth, "invalid depth (n={}!={})".format(
        n, model.zagoruyko_depth)
    return model


image_shape, class_count = ds_train.image_shape, ds_train.class_count
model = get_wide_resnet(
    zaggydepth, k, image_shape, class_count, dim_increase='id')

print("Starting training and validation loop...")
from training import train
train(model, ds_train, ds_val, epoch_count=200)

print("Saving model...")
"""import datetime
model.save_state(dirs.SAVED_MODELS + '/wrn-%d-%d.' % (zaggydepth, k) +
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
"""