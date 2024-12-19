===========================
Submission Data Format
===========================

This document describes the expected format of the data submitted for the CellMap Segmentation Challenge.

Automatic Submission Packaging
------------------------------

For convenience, if you have followed the prediction and processing steps described above and in the example scripts, you can use the following command to zip your predictions in the correct format:

```bash
csc pack-results
```
Additionally, you can explicitly specify the path to the submission zarr, with placeholders {dataset} and {crop}, and the output directory for the zipped submission file using the following command. These default to the PROCESSED_PATH and SUBMISSION_PATH defined in the global configuration file (`config.py`).

The `package_results` function that is used by `csc pack-results` packages a CellMap challenge submission, creating a zarr file, combining all the processed volumes and matching them to the scale and regions of interest of the ground truth crops, and then zipping the result.
    Args:
        input_search_path (str): The base path to the processed volumes, with placeholders for dataset and crops.
        output_path (str | UPath): The path to save the submission zarr to. (ending with `<filename>.zarr`; `.zarr` will be appended if not present, and replaced with `.zip` when zipped).
        overwrite (bool): Whether to overwrite the submission zarr if it already exists.

After packaging the data, they can be visualized alongside the EM, raw predictions, and initial post-processed results using the following command:
```bash
csc visualize -c test -k predictions,processed,submission
```

Structure of the Submission File
--------------------------------

The submission should be a single zip file containing a single Zarr-2 file with the following structure:

```
submission.zarr
    - .zgroup
    - /<test_crop_name>
      - .zgroup
      - /<label_name>
        - .zattrs
        - ...
```

Two options are available for formatting the contents of the label array folder (`/<label_name>/...`):
1. Single scale Zarr-2:
  ```
  - /<label_name>
    - .zattrs - containing "voxel_size"/"resolution"/"scale" and "translation"/"offset"
    - .zarray
    - <...data folders...>
  ```

2. Multiscale OME-NGFF Zarr:

  ```
  - /<label_name>
    - .zattrs - containing "multiscales" metadata
    - .zgroup
    - /s<level> (e.g. "s0")
      - .zarray
      - <...data folders...>
  ```

  An example of multiscales metadata is as follows:
  ```
  {
    "multiscales": [
      {
        "axes": [
            {
                "name": "z",
                "type": "space",
                "unit": "nanometer"
            },
            {
                "name": "y",
                "type": "space",
                "unit": "nanometer"
            },
            {
                "name": "x",
                "type": "space",
                "unit": "nanometer"
            }
        ],
        "datasets": [
          {
            "coordinateTransformations": [
              {
                "scale": [
                    2.0,
                    2.0,
                    2.0
                ],
                "type": "scale"
              },
              {
                "translation": [
                    2760.0,
                    5160.0,
                    10670.0
                ],
                "type": "translation"
              }
            ],
            "path": "s0"
          },
          ...
        ],
        "version": "0.4"
      }
    ]
  }
  ```


The names of the test crops and labels should match the names of the test crops and labels as specified in [the test_crop_manifest](src/cellmap_segmentation_challenge/utils/test_crop_manifest.csv). Similarly, you will see the scale, spatial offset (in nanometers), and shape (in voxels) for each test image. The scale, spatial offset, and shape will automatically be adjusted as necessary during evaluation, if this metadata is present in the `.zattrs` file for each image. Using `csc pack-results` will also do this adjustment for you, allowing you to preview the results of resampling prior to submission (see `evaluation_resampling.rst` for more detailed information). Submitting higher-resolution data will likely lead to the best results after resampling.

Connected Components for Instance Segmentation
----------------------------------------------

Connected components will be run on all instance segmentation submissions to be consistent with the ground truth instance labels. The ground truth instance masks are formed by running connected components on binary semantic masks, which won't necessarily always be correct. Thus, we should ensure the same errors within the submitted data. This means that participants do not need to run instance segmentation specific post-processing on their data prior to submission.

Convenience functions for manual conversion and packaging
---------------------------------------------------------

Assuming each label volume is either:
A) a 3D binary volume with the same shape and scale as the corresponding test volume, or
B) instance IDs per object,
you can convert the submission to the required format using the following convenience functions:

- For converting a single 3D numpy array of class labels to a Zarr-2 file, use the following function:
  `cellmap_segmentation_challenge.utils.evaluate.save_numpy_labels_to_zarr`
Note: The class labels should start from 1, with 0 as background.

- For converting a list of 3D numpy arrays of binary or instance labels to a Zarr-2 file, use the following function:
  `cellmap_segmentation_challenge.utils.evaluate.save_numpy_binary_to_zarr`
Note: The instance labels, if used, should be unique IDs per object, with 0 as background.

The arguments for both functions are the same:
- `submission_path`: The path to save the Zarr-2 file (ending with <filename>.zarr).
- `test_volume_name`: The name of the test volume.
- `label_names`: A list of label names corresponding to the list of 3D numpy arrays or the number of the class labels (0 is always assumed to be background).
- `labels`: A list of 3D numpy arrays of binary labels or a single 3D numpy array of class labels.
- `overwrite`: A boolean flag to overwrite the Zarr-2 file if it already exists.

To zip the Zarr-2 file, you can use the following command:
```bash
zip -r submission.zip submission.zarr
```
