# Deep convolutional networks for semantic segmentation

## Code and data organization
``` latex
semseg
├── storage  % data: datasets, trained models, log-files, ...
├── data  % data loading and preparation
|   ├── preparers
|   |   ├── abstract_preparer.py
|   |   └── iccv09_preparer.py 
|   ├── dataset_dir.py  % helper functions
|   └── dataset.py  % Dataset, MiniBatchReader
├── processing  % image and label processing
|   ├── image_format.py
|   ├── labels.py
|   ├── shape.py  % TODO: test
|   └── transform.py  % TODO: use skimage.transform and test
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
- test `data` more (maybe write some unit tests)
### Medium priority
- write unit tests
- suggest changes
### Low priority
- modify `evaluation.py` so that it makes use of numpy
- test `processing.transform.shape.py`
- fix `processing.transform.py` (use skimage instead of cv2)
