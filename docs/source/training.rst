Training
========

This document covers details involved in training models for the CellMap Segmentation Challenge.

Training configuration
-----------------------

The `csc train <path/to/training_config.py>` functionality loads training configuration parameters from a python file, including constructing the model to be trained.

        <path/to/training_config.py> should be replaced with the path to the configuration file to use for training. Supported settings that will be read from this file if specified include:
        - model_save_path: Path to save the model checkpoints. Default is 'checkpoints/{model_name}_{epoch}.pth'.
        - logs_save_path: Path to save the logs for tensorboard. Default is 'tensorboard/{model_name}'. Training progress may be monitored by running `tensorboard --logdir <logs_save_path>` in the terminal.
        - datasplit_path: Path to the datasplit file that defines the train/val split the dataloader should use. Default is 'datasplit.csv'.
        - validation_prob: Proportion of the datasets to use for validation. This is used if the datasplit CSV specified by `datasplit_path` does not already exist. Default is 0.3.
        - learning_rate: Learning rate for the optimizer. Default is 0.0001.
        - batch_size: Batch size for the dataloader. Default is 8.
        - input_array_info: Dictionary containing the shape and scale of the input data. Default is {'shape': (1, 128, 128), 'scale': (8, 8, 8)}.
        - target_array_info: Dictionary containing the shape and scale of the target data. Default is to use `input_array_info`.
        - epochs: Number of epochs to train the model for. Default is 1000.
        - iterations_per_epoch: Number of iterations per epoch. Each iteration includes an independently generated random batch from the training set. Default is 1000.
        - random_seed: Random seed for reproducibility. Default is 42.
        - classes: List of classes to train the model to predict. This will be reflected in the data included in the datasplit, if generated de novo after calling this script. Default is ['nuc', 'er'].
        - model_name: Name of the model to use. If the config file constructs the PyTorch model, this name can be anything. If the config file does not construct the PyTorch model, the model_name will need to specify which included architecture to use. This includes '2d_unet', '2d_resnet', '3d_unet', '3d_resnet', and 'vitnet'. Default is '2d_unet'. See the `models` module `README.md` for more information.
        - model_to_load: Name of the pre-trained model to load. Default is the same as `model_name`.
        - model_kwargs: Dictionary of keyword arguments to pass to the model constructor. Default is {}. If the PyTorch `model` is passed, this will be ignored. See the `models` module `README.md` for more information.
        - model: PyTorch model to use for training. If this is provided, the `model_name` and `model_to_load` can be any string. Default is None.
        - load_model: Which model checkpoint to load if it exists. Options are 'latest' or 'best'. If no checkpoints exist, will silently use the already initialized model. Default is 'latest'.
        - spatial_transforms: Dictionary of spatial transformations to apply to the training data. Default is {'mirror': {'axes': {'x': 0.5, 'y': 0.5}}, 'transpose': {'axes': ['x', 'y']}, 'rotate': {'axes': {'x': [-180, 180], 'y': [-180, 180]}}}. See the `dataloader` module documentation for more information.
        - validation_time_limit: Maximum time to spend on validation in seconds. If None, there is no time limit. Default is None.
        - validation_batch_limit: Maximum number of validation batches to process. If None, there is no limit. Default is None.
        - device: Device to use for training. If None, will use 'cuda' if available, 'mps' if available, or 'cpu' otherwise. Default is None.
        - use_s3: Whether to stream data from the S3 bucket during training. Default is False.


Model Loading
-------------

The `model_load.py` file contains functions for loading model checkpoints:

- `load_latest(search_path, model)`: Load the latest checkpoint from a directory into a model.
- `load_best_val(logs_save_path, model_save_path, model, low_is_best=True)`: Load the model weights with the best validation score, as recorded in tensorboard logs, from a directory into an existing model object.

For more detailed information about the models and their usage, refer to the [models documentation](../../src/cellmap_segmentation_challenge/models/README.md) and the [model loading documentation](load_model_weights.rst).

Datasplit Generation
---------------------

The datasplit generation process creates a CSV file that defines the train/validation split for the datasets. This is done using the function `make_datasplit_csv` in `src/cellmap_segmentation_challenge/utils/datasplit.py`.

The datasplit generation process involves the following steps:
1. Defines the paths to the raw and groundtruth data and the label classes by crawling the directories and writing the paths to a CSV file.
2. Uses the `glob` function to find the paths to the datasets and store them in a dictionary.
3. Determines the usage (train or validate) for each dataset based on the `validation_prob` parameter.
4. Writes the dataset paths and their usage to the CSV file.

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

Training from S3 Data
---------------------

The `make_s3_datasplit_csv` function in `src/cellmap_segmentation_challenge/utils/datasplit.py` handles the datasplit generation process for S3 data. This function is similar to `make_datasplit_csv`, but it uses S3 (remote) data stores instead of locally downloaded stores. The function `make_s3_datasplit_csv` takes the same parameters as `make_datasplit_csv`. Whether to stream data from s3 during training can be set in the training config file by includeing `use_s3 = True`.

Validation Time/Batch Limit Setting
-----------------------------------

The validation time and batch limit settings allow you to control the maximum time or number of batches to process during validation. These settings are useful for limiting the validation time and ensuring that the validation process does not take too long.

The `validation_time_limit` and `validation_batch_limit` parameters can be set in the configuration file used for training. These parameters are optional and can be set to `None` if there is no time or batch limit.

Examples and Usage Instructions
-------------------------------

Example scripts for training 2D and 3D models are provided in the `examples` directory. The scripts are named `train_2D.py` and `train_3D.py`, respectively, and are thoroughly annotated for clarity. You can run one such script by running the following on the command line:

```bash
cd examples
python train_2D.py
```
This will train a 2D model on the training data and save the model weights to a file (defaults to `./checkpoints/*.pth`), along with logging loss and validation metrics, and sample predictions, viewable in TensorBoard. To view the TensorBoard logs, and monitor training progress, run the following command (assuming you are using the default log directory):

```bash
tensorboard --logdir=tensorboard
```
This will start the TensorBoard server, which you can view in your browser at the link indicated in the terminal, often this is `http://localhost:6006`.

You can also train a 3D model by running the same command with `train_3D.py`:

```bash
python train_3D.py
```

Alternatively, you can use the built-in command to train a model, given the path to a model configuration file. To learn more, you can run the following in the terminal:

```bash
csc train --help
```

For more information on the available options and how training works, see the `README.md` in the `examples` folder, as well as the documentation in `examples/train_2D.py` and `examples/train_3D.py`.
