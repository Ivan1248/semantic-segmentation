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
|   |   ├── layers.py  % conv, max_pool, resize, 
|   |   └── variables.py 
|   ├── abstract_model.py  % AbstractModel
|   └── baseline_a.py  % BaselineA
├── processing  % image and label processing
|   ├── image_format.py
|   ├── labels.py
|   ├── shape.py  % TODO: test resize
|   └── transform.py  % TODO: use skimage.transform and test
├── test  # unit tests
|   ├── test_data.py
|   ├── test_models.py
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
- write more unit tests where needed
- check whether there is a better way of (relative) importing of modules so that they work the same way independent of from what directory they are run from (currently paths are added manually to `sys.path`)
- create `tf_utils.evaluation` and move there accuracy calculation from `AbstractModel` 
- implement evaluation measures used in [FCN](https://arxiv.org/pdf/1411.4038.pdf) and [LinkNet](https://arxiv.org/pdf/1707.03718.pdf) and modify `evaluation.py` so that it makes use of numpy/scipy
- move the accuracy measure from AbstractModel to `tf_utils.evaluation`, add (mean) IoU as well
- add `stride:int` and `dilation:int` parameters to `tf_utils.layers.conv`
#### Medium priority
- create a dummy baseline that assigns each pixel to the most frequent class in the training set (no TensorFlow required)
- add batch normalization to `tf_utils.layers`, use `tf.layers.batch_normalization(input_layer, fused=True, data_format='NCHW')`
- improve random seeding in `Dataset` for beter reproducitibility
- add ResNet layers and encoder/decoder blocks used in LinkNet to `tf_utils.blocks`  
#### Low priority
- improve documentation (and documentation style)
- make a baseline similar to `BaselineA` that uses strided convolutions instead of pooling layers (use 3x3 conv with stride 2 instead of pool->conv)
- add transposed convolution to `tf_utils.layers`
- fix and test `processing.transform.py` (replace `cv2` with `skimage`)
- use polynomial learning rate decay
- make a better baseline
- test and fix `processing.shape` - `resize` isn't tested
#### Work in progress
- waiting for others ~ Ivan
#### Completed
- make data loading work (`data.preparers.Iccv09Preparer, data.Dataset`)
- make a simple baseline
- complete `abstract_model`
- make the cost function (as well as other used evaluation measures) in `BaselineA` ignore "unknown" class (class 0)
- implement `util.Visualizer` 
- improve the colors in `util.visualizer.Visualizer`

## Current validation results on _Stanford Background Dataset_
Model                 | mIoU | Pixel acc. | #epochs | Infer. time [s] | Hardware        |
--------------------- | ----:| ----------:| -------:| ---------------:| ----------------|
BaselineA(lr:4e-4)    | -    |      0.603 |     150 | (mb16)  ~<0.100 | Pentium 2020M   |

_"Inference time" - on what hardware?_