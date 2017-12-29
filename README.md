# Deep convolutional networks for semantic segmentation

Parts of the code are are based on [this project](https://bitbucket.org/Ivan1248/semantic-image-segmentation-by-deep-convolutional-networks/).

## Code and data organization (outdated)
``` tex
semseg
├── storage  % data (not code): datasets, trained models, log-files, ...
├── data  % data loading and preparation: Dataset, MiniBatchReader
|   ├── preparers  % Iccv09Preparer
|   |   ├── abstract_preparer.py
|   |   └── iccv09_preparer.py 
|   ├── dataset_dir.py
|   └── dataset.py
├── models  % AbstractModel, BaselineA
|   ├── preprocessing.py
|   ├── tf_utils
|   |   ├── blocks.py  % higher-level operations: ResNet block, LinkNet rencoder/decoder blocks 
|   |   ├── layers.py  % elementary operations: conv, max_pool, resize
|   |   └── variables.py 
|   ├── abstract_model.py  % AbstractModel
|   └── baseline_a.py  % BaselineA
├── processing  % image and label processing
|   ├── image_format.py
|   ├── labels.py
|   ├── shape.py  % TODO: test resize
|   └── transform.py  % TODO: use skimage.transform and test
├── test  % unit tests
|   :.
├── util  % helper functions and classes
|   ├── visualizer.py  % Visualizer
|   :.
├── evaluation.py
:.
```

## How to contribute
1. decide that you don't want to feel bad later
2. look below

## Tasks
#### High priority
- implement evaluation measures used in [FCN](https://arxiv.org/pdf/1411.4038.pdf) and [LinkNet](https://arxiv.org/pdf/1707.03718.pdf) and modify `evaluation.py` so that it makes use of numpy/scipy
- evaluate
- implement inference time measurement depending on mini-batch size (to compare with the results in the LinkNet paper)
- fix model saving and loading
- write the report
#### Medium priority
- nothing
#### Low priority
- use `tf.nn.sparse_softmax_cross_entropy_with_logits` for more efficient training
- make a baseline similar to `BaselineA` that uses strided convolutions instead of pooling layers (use 3x3 conv with stride 2 instead of pool->conv)
- try IoU loss (like [here](http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html))
#### Work in progress
- almost everything ~ Ivan
- implement LinkNet ~ Josip
#### Completed
- make data loading work (`data.preparers.Iccv09Preparer, data.Dataset`)
- make a simple baseline
- complete `abstract_model`
- make the cost function (as well as other used evaluation measures) in `BaselineA` ignore "unknown" class (class 0)
- implement `util.Visualizer` 
- improve the colors in `util.visualizer.Visualizer`
- enable usage of `util.Visualizer` while training (by pressing _d_ followed by _ENTER_ in the console)
- add `stride:int` and `dilation:int` parameters to `tf_utils.layers.conv` (use `tf.nn.convolution`)
- add batch normalization to `tf_utils.layers`, use `tf.layers.batch_normalization(input_layer, fused=True, data_format='NCHW')`
- improve random seeding in `Dataset` for beter reproducibility
- add ResNet layers and encoder/decoder blocks used in LinkNet to `tf_utils.blocks`
- implement a textual options menu that can be opened while training is paused, enabling network output visualization, saving/loading of weights, stopping training (after the current epoch) and other actions
- add transposed convolution to `tf_utils.layers`

## Current validation results on _Stanford Background Dataset_
mini-batch size = 16, Pentium 2020M

Model        | mIoU | Pixel acc. | #epochs  | Inference time [s] | Hardware        |
------------ | ----:| ----------:| --------:| ------------------:| ----------------|
BaselineA    | -    |      0.630 |      150 |             0.100* | Pentium 2020M   |

_"Inference time" - on what hardware?_
