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
├── models  % AbstractModel, BaseelineA
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
├── util  # helper functions and classes
|   ├── directory.py  % get_files
|   ├── file.py  % read_all_text, read_all_lines, write_all_text 
|   ├── path.py  % get_file_name, get_file_name_without_extension
|   ├── visualizer.py  % Visualizer instead of cv2
|   :.
├── evaluation.py
:.
```

## How to contribute
- pick one or more tasks (move it to _Work in progress_ and add "~" followed by your name(s))
- improve some part of code
- write some more unit tests
- improve documentation (and documentation style)
- modify this README if you have some idea: add, remove or modify any section as you like

## Tasks
#### High priority
- write more unit tests where needed
- improve `models.BaselineA`
- check whether there is a better way of (relative) importing of modules so that they work the same way independent of from what directory they are run from (currently paths are added manually to `sys.path`)
- move the `train` method to `AbstractModel`?
- add a `test` metohod to `BaselineA` or `AbstractModel`
- implement evaluation measures used in [FCN](https://arxiv.org/pdf/1411.4038.pdf) and [LinkNet](https://arxiv.org/pdf/1707.03718.pdf) and modify `evaluation.py` so that it makes use of numpy/scipy
- make `Dataset` bahavior reproducible/deterministic
#### Medium priority
- create a dummy baseline that assigns each pixel to the most frequent class in the training set (no TensorFlow required)
- add batch normalization to `tf_utils.layers`
- add `stride:int` parameter to `tf_utils.layers.conv`
- test and fix `processing.shape` - `resize` isn't tested
#### Low priority
- make a baseline similar to `BaselineA` that uses strided convolutions instead of pooling layers (use 3x3 conv with stride 2 instead of pool->conv)
- add `dilation` parameter to `tf_utils.layers.conv`
- add transposed convolution to `tf_utils.layers`
- fix and test `processing.transform.py` (replace `cv2` with `skimage`)
#### Work in progress
- make `util.Visualizer` work ~ Ivan
#### Completed
- make the cost function (as well as other used evaluation measures) in `BaselineA` ignore "unknown" class (class 0)
- complete `abstract_model`
- make data loading work (`data.preparers.Iccv09Preparer, data.Dataset`)
- make a simple baseline (accuracy 0.4 after 15 epochs (32 minutes on CPU))


## Current validation results on _Stanford Background Dataset_
Model     | mIoU | Pixel accuracy | Epoch count | Inference time<sup>[*](#myfootnote1)</sup>
--------- | ---- | -------------- | -----------:| -
BaselineA | -    | 0.400          | 15          | ?

<a name="myfootnote1">*</a>on what hardware?