# CellMap Segmentation Challenge Examples

This directory contains examples for the CellMap Segmentation Challenge. The examples include:
1. Training 2D and 3D models
2. Predicting on test data
3. Evaluating predictions

## Training 2D and 3D models
The `train_2D.py` and `train_3D.py` scripts train 2D and 3D models, respectively, on the CellMap Segmentation Challenge dataset. The scripts use a configuration file, which defines the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file path as an argument to start the training process. The `train` function reads the configuration file, sets up the data loaders, model, optimizer, loss function, and other components, and trains the model for the specified number of epochs.

The configuration file includes the following components:
1. Hyperparameters: learning rate, batch size, input and target array information, epochs, iterations per epoch, and random seed.
2. Model: model architecture, number and type of classes, and other model-specific configurations. This should return a PyTorch model.
3. Paths: paths for saving logs, model checkpoints, and data split file.
4. Spatial transformations: spatial augmentations to apply to the training data.

These configuration files can then be used to run training via either one of two commands:
1. `python path/to/train_config.py`: Run the training script directly.
2. `csc train path/to/train_config.py`: Run the training script using the `csc train` command-line interface.

For example, to train a 3D model using the configuration file `train_3D.py`, you can run the following command from the `examples` directory:

```bash
csc train train_3D.py
```

Training progress can be monitored using TensorBoard by running `tensorboard --logdir tensorboard` in the terminal.

Once the model is trained, you can use the `predict` function to make predictions on new data using the trained model. See the `predict_3D.py` and `predict_2D.py` scripts (and below) for examples of how to use the `predict` function.

### Datasplit Generation

The datasplit generation process creates a CSV file that defines the train/validation split for the datasets. This is done using the function `make_datasplit_csv` in `src/cellmap_segmentation_challenge/utils/datasplit.py`.

The datasplit generation process involves the following steps:
1. Define the paths to the raw and groundtruth data and the label classes by crawling the directories and writing the paths to a CSV file.
2. Use the `glob` function to find the paths to the datasets and store them in a dictionary.
3. Determine the usage (train or validate) for each dataset based on the `validation_prob` parameter.
4. Write the dataset paths and their usage to the CSV file.

The function `make_datasplit_csv` takes the following parameters:
- `classes`: A list of classes to include in the CSV.
- `force_all_classes`: A boolean flag to force all classes to be present in the training/validation datasets.
- `validation_prob`: The probability of a dataset being in the validation set.
- `datasets`: A list of datasets to include in the CSV.
- `crops`: A list of crops to include in the CSV.
- `search_path`: The search path to use to find the datasets.
- `raw_name`: The name of the raw data.
- `crop_name`: The name of the crop.
- `csv_path`: The path to write the CSV file to.
- `dry_run`: A boolean flag to perform a dry run without writing the CSV file.

For more detailed information, refer to the [datasplit generation documentation](../docs/source/dataloader.rst).

### Training from S3 Data

The `make_s3_datasplit_csv` function in `src/cellmap_segmentation_challenge/utils/datasplit.py` handles the datasplit generation process for S3 data. This function is similar to `make_datasplit_csv`, but it uses S3 (remote) data stores instead of locally downloaded stores.

The function `make_s3_datasplit_csv` takes the same parameters as `make_datasplit_csv`, with the addition of the `use_s3` parameter to specify whether to use S3 data stores.

For more detailed information, refer to the [datasplit generation documentation](../docs/source/dataloader.rst).

### Validation Time/Batch Limit Setting

The validation time and batch limit settings allow you to control the maximum time and number of batches to process during validation. These settings are useful for limiting the validation time and ensuring that the validation process does not take too long.

The `validation_time_limit` and `validation_batch_limit` parameters can be set in the configuration file used for training. These parameters are optional and can be set to `None` if there is no time or batch limit.

For more detailed information, refer to the [training documentation](../docs/source/dataloader.rst).

## Predicting on test data
The `predict_2D.py` and `predict_3D.py` scripts demonstrate how to use a trained model to make predictions on test data. The predictions are saved as Zarr-2 files in the specified output directory. The scripts use a configuration file to define model and other configurations required for making predictions, this file can be the same used for training the model. The scripts call the `predict` function with the path to this configuration file as an argument. For example, to predict on the test data using the 3D model from `train_3D.py`, you can do so directly by running the following command:

```bash
csc predict train_3D.py
```

To see the other options available for the `predict` command, such as picking crops to predict on or setting an output path, you can run `csc predict --help`.

## Post-processing predictions
The `process_2D.py` and `process_3D.py` configuration scripts demonstrate how to post-process the predictions made by a model. Examples of post-processing steps include thresholding, merging IDs for connected components, filtering based on object size, etc.. The scripts define the post-processing parameters, including the `input_array_info` and `target_array_info` of the processing (same as in the training configuration), which `classes` to process, `batch_size` for dataloading, which `crops` to process (or "test" to process all test crops), and the post-processing function to apply. The scripts, when run with python, call the `process` function with the path to this configuration file as an argument. For example, to post-process the predictions made by the 3D model from `train_3D.py`, you can do so directly by running the following command:

```bash
python process_3D.py
```

Or, to run the post-processing script with the `csc` command-line interface, you can run the following command:

```bash
csc process process_3D.py
```

To see the other options available for the `process` command, you can run `csc process --help`.

During evaluations of submissions, for instance segmentation evaluated classes, connected components are computed on the supplied masks and the resulting instance IDs are assigned to each connected component. This will not merge already uniquely IDed objects. Thus, you do not need to run connected components on before submission, but you may wish to execute more advanced post-processing for instance segmentation, such as watershed.

## Visualizing data and predictions

You can visualize the data and predictions using the `visualize.py` module. This module provides functions to visualize the data and predictions using neuroglancer. To see the available options, see [the documentation](../docs/source/visualization.rst) or run the following command:

```bash
csc visualize --help
```
To submit your predictions, first make sure that they are in the correct format (see below), then submit them through [the online submission portal](https://staging.cellmapchallenge.2i2c.cloud/upload). You will need to sign in with your GitHub account to submit your predictions.

## Data submission

For convenience, if you have followed the prediction and processing steps described above and in the example scripts, you can use the following command to zip your predictions in the correct format:

```bash
csc pack-results
```
Additionally, you can explicitly specify the path to the submission zarr, with placeholders {dataset} and {crop}, and the output directory for the zipped submission file using the following command. These default to the PROCESSED_PATH and SUBMISSION_PATH defined in the global configuration file (`config.py`).

### Evaluation Resampling

Evaluation resampling ensures that the predicted and ground truth volumes are compared at the same resolution and shape. This is crucial for accurate evaluation of the model's performance.

The resampling process involves adjusting the resolution and shape of the predicted volumes to match those of the ground truth volumes. This is done using different interpolation methods depending on the type of segmentation:

- **Instance Segmentations**: Nearest neighbor interpolation is used to preserve the discrete nature of instance labels.
- **Semantic Segmentations**: Linear interpolation is used to smoothly adjust the continuous nature of semantic labels.

The function `match_crop_space` in `src/cellmap_segmentation_challenge/evaluate.py` handles the resampling process. It takes the following parameters:

- `path`: The path to the zarr array to match.
- `class_label`: The class label of the array.
- `voxel_size`: The target voxel size.
- `shape`: The target shape.
- `translation`: The translation (i.e., offset) of the array in world units.

For more detailed information, refer to the [evaluation resampling documentation](../docs/source/evaluation_resampling.rst).

### Manual data packaging
If you are packaging your predictions manually, the submission file format requirements are as follows:

Submission file format requirements:
1. The submission should be a single zip file containing a single Zarr-2 file with the following structure:
```
submission.zarr
    - /<test_volume_name>
    - /<label_name>
```
2. The names of the test volumes and labels should match the names of the test volumes and labels in the test data. See `examples/predict_2D.py` and `examples/predict_3D.py` for examples of how to generate predictions in the correct format.
3. The scale for all volumes is 8x8x8 nm/voxel, except as otherwise specified.

Assuming your data is already 8x8x8nm/voxel,and each label volume is either A) a 3D binary volume with the same shape and scale as the corresponding test volume, 
or B) instance IDs per object, you can convert the submission to the required format using the following convenience functions:

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
