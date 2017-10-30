# Deep convolutional networks for semantic segmentation

Parts of the code are are based on [this project](https://bitbucket.org/Ivan1248/semantic-image-segmentation-by-deep-convolutional-networks/).

## Code and data organization
``` tex
semseg
├── data  % data loading and preparation
|   ├── preparers
|   |   ├── abstract_preparer.py
|   |   └── iccv09_preparer.py 
|   ├── dataset_dir.py  % helper functions
|   └── dataset.py  % Dataset, MiniBatchReader
├── models
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
├── storage  % data (not code): datasets, trained models, log-files, ...
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
|   ├── display_window.py  % DisplayWindow, TODO: use matplotlib instead of cv2
|   :.
├── evaluation.py
:.
```

## Tasks
#### Higher priority
- write more tests
- suggest/make improvements
- improve `models.BaselineA`
- check whether there is a better way of (relative) importing of modules so that they work the same way independent of from what directory they are run from (currently paths are added manually to `sys.path`)
- make `util.ResultDisplayWindow` work 
- add `stride` parameter to `tf_utils.layers.conv`
#### Lower priority
- add `dilation` parameter to `tf_utils.layers.conv`
- add transposed convolution to `tf_utils.layers`
- add batch normalization to `tf_utils.layers`
- modify `evaluation.py` so that it makes use of numpy
- test and fix `processing.shape` - `resize` isn't tested
- fix and test `processing.transform.py` (replace `cv2` with `skimage`)
#### Maybe completed
- finish `abstract_model`
#### Completed
- make a working baseline
- test and fix data loading (`data.preparers.Iccv09Preparer, data.Dataset}`)
