Evaluation Resampling
=====================

Purpose
-------
Evaluation resampling ensures that the predicted and ground truth volumes are compared at the same resolution and region of interest (ROI) with the dataset. This is crucial for accurate evaluation of the model's performance.

Resampling Process
------------------
The resampling process involves adjusting the resolution and ROI of the predicted volumes to match those of the ground truth volumes. This is done using different interpolation methods depending on the type of segmentation:

- **Instance Segmentations**: Nearest neighbor interpolation is used to preserve the unique IDs of instance labels.
- **Semantic Segmentations**: Linear interpolation followed by thresholding is used, as it is more accurate than nearest neighbor interpolation.

Note: Linear interpolation followed by thresholding can also be used for instance segmentations, but requires doing so iteratively for each unique ID. We do not implement this here.

Function
--------
The function `match_crop_space` in `src/cellmap_segmentation_challenge/evaluate.py` handles the resampling process. It takes the following parameters:

- `path` (str | UPath): The path to the zarr array to be adjusted. The zarr can be an OME-NGFF multiscale zarr file, or a traditional single scale formatted zarr.
- `class_label` (str): The class label of the array.
- `voxel_size` (tuple): The target voxel size.
- `shape` (tuple): The target shape.
- `translation` (tuple): The translation (i.e. offset) of the array in world units.
