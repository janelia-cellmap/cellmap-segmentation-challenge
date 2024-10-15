# CellMap Segmentation Challenge Examples

This directory contains examples for the CellMap Segmentation Challenge. The examples include:
1. Training 2D and 3D models
2. Predicting on test data
3. Evaluating predictions

## Training 2D and 3D models
The `train_2D.py` and `train_3D.py` scripts train 2D and 3D models, respectively, on the CellMap Segmentation Challenge dataset. The scripts use a configuration file to define the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file as an argument to start the training process.

The configuration file defines the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file as an argument to start the training process. The `train` function reads the configuration file, sets up the data loaders, model, optimizer, loss function, and other components, and trains the model for the specified number of epochs.

The configuration file includes the following components:
1. Hyperparameters: learning rate, batch size, input and target array information, epochs, iterations per epoch, random seed, and initial number of features for the model.
2. Model: model architecture, number of classes, and other model-specific configurations. This should return a PyTorch model.
3. Paths: paths for saving logs, model checkpoints, and data split file.
4. Spatial transformations: spatial transformations to apply to the training data.

These configuration files can then be used to run training via two different commands:
1. `python path/to/train_config.py`: Run the training script directly.
2. `csc train path/to/train_config.py`: Run the training script using the `csc train` command-line interface.

Training progress can be monitored using TensorBoard by running `tensorboard --logdir tensorboard` in the terminal.

Once the model is trained, you can use the `predict` function to make predictions on new data using the trained model. See the `predict_3D.py` and `predict_2D.py` scripts (and below) for examples of how to use the `predict` function.

## Predicting on test data
...

## Submission requirements:
1. The submission should be a single zip file containing a single Zarr-2 file with the following structure:
   - submission.zarr
     - /<test_volume_name>
        - /<label_name>
2. The names of the test volumes and labels should match the names of the test volumes and labels in the test data.
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
`zip -r submission.zip submission.zarr`

To submit the zip file, upload it to the challenge platform.
