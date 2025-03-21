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

### Training from S3 Data
The `make_s3_datasplit_csv` function in `src/cellmap_segmentation_challenge/utils/datasplit.py` handles the datasplit generation process for S3 data. This function is similar to `make_datasplit_csv`, but it uses S3 (remote) data stores instead of locally downloaded stores. The function `make_s3_datasplit_csv` takes the same parameters as `make_datasplit_csv`. Without preconstructing a datasplit csv file you can direct training to stream data from S3 by including `use_s3 = True` in your training configuration file.

### Validation Time/Batch Limit Setting

The validation time and batch limit settings allow you to control the maximum time and number of batches to process during validation. These settings are useful for limiting the validation time and ensuring that the validation process does not take too long.

The `validation_time_limit` and `validation_batch_limit` parameters can be set in the configuration file used for training. These parameters are optional and can be set to `None` if there is no time or batch limit.

For more detailed information, refer to the [training documentation](../docs/source/dataloader.rst).

## Predicting on test data
The `predict_2D.py` and `predict_3D.py` scripts demonstrate how to use a trained model to make predictions on test data, and can be used by running `python predict_2D.py` and `python predict_3D.py`, respectively. The predictions are saved as Zarr-2 files in the specified output directory. The scripts use a configuration file to define model and other configurations required for making predictions, this file can be the same used for training the model. The scripts call the `predict` function with the path to this configuration file as an argument. For example, to predict on the test data using the 3D model from `train_3D.py`, you can do so directly by running the following command:

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

Evaluation resampling ensures that the predicted and ground truth volumes are compared at the same resolution and region of interest (ROI). This is crucial for accurate evaluation of the model's performance. The resampling process adjusts the resolution and ROI of the predicted volumes to match those of the ground truth volumes. For more detailed information, refer to the [evaluation resampling documentation](../docs/source/evaluation_resampling.rst).

### Data Format

For more detailed information on the expected format of the data submitted, refer to the [submission_data_format.rst](../docs/source/submission_data_format.rst) file.

### Extended Training Configuration
The training pipeline (`train.py`) accepts a configuration file that defines the training parameters, model architecture, and other settings. Here is the current list of included parameters that can be set in the configuration file:
- **batch_size**: Batch size for the dataloader. Default is `8`.
- **classes**: List of classes to train the model to predict. This will be reflected in the data included in the datasplit, if generated de novo after calling this script. Default is `['nuc', 'er']`.
- **criterion**: Uninstantiated PyTorch loss function to use for training. Default is `torch.nn.BCEWithLogitsLoss`.
- **criterion_kwargs**: Dictionary of keyword arguments to pass to the loss function constructor. Default is `{}`.
- **datasplit_path**: Path to the datasplit file that defines the train/val split the dataloader should use. Default is `'datasplit.csv'`.
- **device**: Device to use for training. If `None`, will use `'cuda'` if available, `'mps'` if available, or `'cpu'` otherwise. Default is `None`.
- **epochs**: Number of epochs to train the model for. Default is `1000`.
- **filter_by_scale**: Whether to filter the data by scale. If `True`, only data with a scale less than or equal to the `input_array_info` highest resolution will be included in the datasplit. If set to a scalar value, data will be filtered for that isotropic resolution - anisotropic can be specified with a sequence of scalars. Default is `False` (no filtering).
- **force_all_classes**: Whether to force all classes to be present in each batch provided by dataloaders. Can either be `True` to force this for both validation and training dataloader, `False` to force for neither, or `train` / `validate` to restrict it to training or validation, respectively. Default is `'validate'`.
- **gradient_accumulation_steps**: Number of gradient accumulation steps to use. Default is `1`. This can be used to simulate larger batch sizes without increasing memory usage.
- **input_array_info**: Dictionary containing the shape and scale of the input data. Default is `{'shape': (1, 128, 128), 'scale': (8, 8, 8)}`.
- **iterations_per_epoch**: Number of iterations per epoch. Each iteration includes an independently generated random batch from the training set. Default is `1000`.
- **learning_rate**: Learning rate for the optimizer. Default is `0.0001`.
- **load_model**: Which model checkpoint to load if it exists. Options are `'latest'` or `'best'`. If no checkpoints exist, will silently use the already initialized model. Default is `'latest'`.
- **logs_save_path**: Path to save the logs for tensorboard. Default is `'tensorboard/{model_name}'`. Training progress may be monitored by running `tensorboard --logdir <logs_save_path>` in the terminal.
- **max_grad_norm**: Maximum gradient norm for clipping. If `None`, no clipping is performed. Default is `None`. This can be useful to prevent exploding gradients which would lead to NaNs in the weights.
- **model**: PyTorch model to use for training. If this is provided, the `model_name` and `model_to_load` can be any string. Default is `None`.
- **model_kwargs**: Dictionary of keyword arguments to pass to the model constructor. Default is `{}`. If the PyTorch `model` is passed, this will be ignored. See the `models` module `README.md` for more information.
- **model_name**: Name of the model to use. If the config file constructs the PyTorch model, this name can be anything. If the config file does not construct the PyTorch model, the model_name will need to specify which included architecture to use. This includes `'2d_unet'`, `'2d_resnet'`, `'3d_unet'`, `'3d_resnet'`, and `'vitnet'`. Default is `'2d_unet'`. See the `models` module `README.md` for more information.
- **model_save_path**: Path to save the model checkpoints. Default is `'checkpoints/{model_name}_{epoch}.pth'`.
- **model_to_load**: Name of the pre-trained model to load. Default is the same as `model_name`.
- **optimizer**: PyTorch optimizer to use for training. Default is `torch.optim.RAdam(model.parameters(), lr=learning_rate, decoupled_weight_decay=True)`.
- **random_seed**: Random seed for reproducibility. Default is `42`.
- **scheduler**: PyTorch learning rate scheduler (or uninstantiated class) to use for training. Default is `None`. If provided, the scheduler will be called at the end of each epoch.
- **scheduler_kwargs**: Dictionary of keyword arguments to pass to the scheduler constructor. Default is `{}`. If `scheduler` instantiation is provided, this will be ignored.
- **spatial_transforms**: Dictionary of spatial transformations to apply to the training data. Default is `{'mirror': {'axes': {'x': 0.5, 'y': 0.5}}, 'transpose': {'axes': ['x', 'y']}, 'rotate': {'axes': {'x': [-180, 180], 'y': [-180, 180]}}}`. See the `dataloader` module documentation for more information.
- **target_array_info**: Dictionary containing the shape and scale of the target data. Default is to use `input_array_info`.
- **target_value_transforms**: Transform to apply to the target values. Default is `T.Compose([T.ToDtype(torch.float), Binarize()])` which converts the input masks to float32 and threshold at 0 (turning object ID's into binary masks for use with binary cross entropy loss). This can be used to specify other targets, such as distance transforms.
- **train_raw_value_transforms**: Transform to apply to the raw values for training. Defaults to `T.Compose([T.ToDtype(torch.float, scale=True), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})])` which normalizes the input data, converts it to float32, and replaces NaNs with 0. This can be used to add augmentations such as random erasing, blur, noise, etc.
- **use_mutual_exclusion**: Whether to use mutual exclusion to infer labels for unannotated pixels. Default is `False`.
- **use_s3**: Whether to use the S3 bucket for the datasplit. Default is `False`.
- **val_raw_value_transforms**: Transform to apply to the raw values for validation, similar to `train_raw_value_transforms`. Default is the same as `train_raw_value_transforms`.
- **validation_batch_limit**: Maximum number of validation batches to process. If `None`, there is no limit. Default is `None`.
- **validation_prob**: Proportion of the datasets to use for validation. This is used if the datasplit CSV specified by `datasplit_path` does not already exist. Default is `0.15`.
- **validation_time_limit**: Maximum time to spend on validation in seconds. If `None`, there is no time limit. Default is `None`.
- **weighted_sampler**: Whether to use a sampler weighted by class counts for the dataloader. Default is `True`.
- **weight_loss**: Whether to weight the loss function by class counts found in the datasets. Default is `True`.
