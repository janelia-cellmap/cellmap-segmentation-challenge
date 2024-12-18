# Cellmap Segmentation Challenge Data
Assuming you follow the instructions in the main README.md, this is where your training data should be stored. 

The data will be stored in the following structure:

```
.
├── <dataset name>
│   └── <dataset name>.zarr
│       └── recon-<number>
│           ├── em
│           │   └── fibsem-uint8
│           │       ├── s0 <-- Highest resolution scale level
│           │       ├── s1
│           │       └── ...
│           └── labels
│               └── groundtruth
│                   └── crop<number>
│                       ├── <label class 1>
│                       │   └── s0 <-- Highest resolution
│                       ├── <label class 2>
│                       ├── ...
│                       └── all <-- All labels combined
└── README.md
```

## Test Data
The raw (EM) test data will be stored alongside the training data. In case the regions of interest used for testing change, you will need to update the test data accordingly. If you are using the default data path, you can simply run the following from the root of the repository to do this:

```bash
csc fetch-data --crops test
```

## Data Format
To learn more about the data format, please refer to the [data format documentation](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/) in our AWS S3 bucket [janelia-cosem-datasets](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/).

## Visualization Tool
The `visualize.py` module provides functions to visualize, with neuroglancer, raw (EM) and ground truth data alongside your predictions, post-processed outputs, and results packaged for submission.

To visualize the data and predictions, you can use the `csc visualize` command. This command serves the image arrays on a local server and opens a browser window with the neuroglancer viewer. You can then navigate through the data and predictions and compare them side by side.

For more detailed information, refer to the [visualization documentation](../docs/source/visualization.rst).

## Evaluation Resampling

Evaluation resampling ensures that the predicted and ground truth volumes are compared at the same resolution and region of interest (ROI). This is crucial for accurate evaluation of the model's performance.

The resampling process involves adjusting the resolution and ROI of the predicted volumes to match those of the ground truth volumes. This is done using different interpolation methods depending on the type of segmentation:

- **Instance Segmentations**: Nearest neighbor interpolation is used to preserve the unique IDs of instance labels.
- **Semantic Segmentations**: Linear interpolation followed by thresholding is used to resample semantic labels.

The function `match_crop_space` in `src/cellmap_segmentation_challenge/evaluate.py` handles the resampling process. It takes the following parameters:

- `path`: The path to the zarr array to match.
- `class_label`: The class label of the array.
- `voxel_size`: The target voxel size.
- `shape`: The target shape.
- `translation`: The translation (i.e., offset) of the array in world units.

For more detailed information, refer to the [evaluation resampling documentation](../docs/source/evaluation_resampling.rst).
