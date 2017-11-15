# Deep convolutional networks for semantic segmentation

Parts of the code are are based on [this project](https://bitbucket.org/Ivan1248/semantic-image-segmentation-by-deep-convolutional-networks/).

## Code and data organization
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
|   ├── test_data.py
|   ├── test_processing.py
|   ├── test_util.py
|   :.
├── util  % helper functions and classes
|   ├── directory.py  % get_files
|   ├── file.py  % read_all_text, read_all_lines, write_all_text 
|   ├── path.py  % get_file_name, get_file_name_without_extension
|   ├── visualizer.py  % Visualizer
|   :.
├── evaluation.py
:.
```

## How to contribute
- pick one or more tasks (move it to _Work in progress_ and add "~" followed by your name(s))
- modify this README whenever you think something should be changed/added/removed

## Tasks
#### High priority
- check whether there is a better way of (relative) importing of modules so that they work the same way independent of from what directory they are run from (currently paths are added manually to `sys.path`)
- move the accuracy measure from AbstractModel to `tf_utils.evaluation`, add (mean) IoU as well
- implement inference time measurement depending on mini-batch size
- improve model saving (make more use of `tf.train.Saver`)
#### Medium priority
- add batch normalization to `tf_utils.layers`, use `tf.layers.batch_normalization(input_layer, fused=True, data_format='NCHW')`
- improve random seeding in `Dataset` for beter reproducibility
- add ResNet layers and encoder/decoder blocks used in LinkNet to `tf_utils.blocks`
- implement a textual options menu that can be opened while training is paused, enabling network output visualization, saving/loading of weights, stopping training (after the current epoch) and other actions
- use `tf.nn.sparse_softmax_cross_entropy_with_logits` for more efficient training
- make `Dataset` not shuffle the images, but an array of indexes so that the order can be reset
#### Low priority
- create a dummy baseline that assigns each pixel to the most frequent class in the training set (no TensorFlow required)
- improve documentation (and documentation style)
- make a baseline similar to `BaselineA` that uses strided convolutions instead of pooling layers (use 3x3 conv with stride 2 instead of pool->conv)
- add transposed convolution to `tf_utils.layers`
- fix and test `processing.transform.py` (replace `cv2` with `skimage`)
- use polynomial learning rate decay
- test and fix `processing.shape` - `resize` isn't tested
- try IoU loss (like [here](http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html))
#### Work in progress
- nothing specific ~ Ivan
- implement evaluation measures used in [FCN](https://arxiv.org/pdf/1411.4038.pdf) and [LinkNet](https://arxiv.org/pdf/1707.03718.pdf) and modify `evaluation.py` so that it makes use of numpy/scipy ~ Annie

#### Completed
- make data loading work (`data.preparers.Iccv09Preparer, data.Dataset`)
- make a simple baseline
- complete `abstract_model`
- make the cost function (as well as other used evaluation measures) in `BaselineA` ignore "unknown" class (class 0)
- implement `util.Visualizer` 
- improve the colors in `util.visualizer.Visualizer`
- enable usage of `util.Visualizer` while training (by pressing _d_ followed by _ENTER_ in the console)
- add `stride:int` and `dilation:int` parameters to `tf_utils.layers.conv` (use `tf.nn.convolution`)

## Current validation results on _Stanford Background Dataset_
Model        | mIoU | Pixel acc. | #epochs  | Infer. time [s] | Hardware        |
------------ | ----:| ----------:| --------:| ---------------:| ----------------|
BaselineA    | -    |      0.630 |      150 |          0.100* | Pentium 2020M   |
*mini-batch size = 16, Pentium 2020M

_"Inference time" - on what hardware?_