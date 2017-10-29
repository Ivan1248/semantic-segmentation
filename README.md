# Deep convolutional networks for semantic segmentation

Parts of the code are are based on [this project](https://bitbucket.org/Ivan1248/semantic-image-segmentation-by-deep-convolutional-networks/).

## Code and data organization
``` latex
semseg
├── data  % data loading and preparation
|   ├── preparers
|   |   ├── abstract_preparer.py
|   |   └── iccv09_preparer.py 
|   ├── dataset_dir.py  % helper functions
|   └── dataset.py  % Dataset, MiniBatchReader
├── models
|   ├── abstract_model.py  % AbstractModel
|   └── baseline_a.py  % BaselineA
├── processing  % image and label processing
|   ├── image_format.py
|   ├── labels.py
|   ├── shape.py  % TODO: test
|   └── transform.py  % TODO: use skimage.transform and test
├── storage  % data (not code): datasets, trained models, log-files, ...
├── util  # helper functions and classes
|   ├── directory.py  % get_files
|   ├── file.py
|   ├── path.py
|   ├── display_window.py  % DisplayWindow
|   :.
├── evaluation.py
├── preprocessing.py
├── tf_utils.py
:.
```

## TODOs
### High priority
- finish `abstract_model`
- make a baseline
- write more tests
### Medium priority
- write unit tests
- suggest changes
### Low priority
- modify `evaluation.py` so that it makes use of numpy
- test `processing.transform.shape.py`
- fix `processing.transform.py` (use skimage instead of cv2)
