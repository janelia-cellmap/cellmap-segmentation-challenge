<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# CellMap Segmentation Challenge
Welcome to Cellmap Segmentation Challenge's documentation! This is a Python package for the Cellmap Segmentation Challenge. It provides a Python API for interacting with the challenge data, running training, prediction, and evaluation. The package is built on top of the `cellmap-data <https://github.com/janelia-cellmap/cellmap-data>`_ package, which provides a Python API for interacting with the Cellmap data format. The package is designed to be easy to use, flexible, and extensible.

# Getting started

## Set up your environment

We recommend using micromamba to create a new environment with the required dependencies. You can install micromamba by following the instructions [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install). The fastest way to do this install is to run the following in your terminal:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Once you have micromamba installed, you can create a new environment with the required dependencies by running the following commands:

```bash
# Create a new environment
micromamba create -n cellmap-segmentation-challenge -y python=3.11

# Activate the environment
micromamba activate cellmap-segmentation-challenge
```

## Clone the repository

You can clone and install the repository by running the following command:

```bash
# Clone the repository
git clone https://github.com/janelia-cellmap/cellmap-segmentation-challenge

# Install the repo in editable mode
cd cellmap-segmentation-challenge
pip install -e .
```

## Repository structure

The repository is structured as follows:

```

cellmap-segmentation-challenge/
│
├── examples/
│   ├── train_2d.py
│   ├── train_3d.py
│   ├── predict_2d.py
│   ├── predict_3d.py
│   ├── ...
│   └── README.md
│
├── src/
│   ├── cellmap_segmentation_challenge/
│   │   ├── models/
│   │   │   ├── model_load.py
│   │   │   ├── unet_model_2D.py
│   │   │   ├── ...
│   │   │   └── README.md
│   │   └── ...
│
├── data/
│   ├── ...
│   └── README.md
│
├── ...
│
├── README.md
└── ...
```



## Download the data

Once you have installed this package, you can download the challenge data by running the following command:

```bash
csc fetch-data --crops all --dest data
```

This will retrieve groundtruth data and corresponding EM data for each crop and save it at a path 
of your choosing on your local filesystem.

To see all the options for the `fetch-data` command, run 

```bash
csc fetch-data --help
```

## Train a model

Example scripts for training 2D and 3D models are provided in the `examples` directory. The scripts are named `train_2d.py` and `train_3d.py`, respectively, and are thoroughly annotated for clarity. You can run one such script by running the following on the command line:

```bash
cd examples
python train_2d.py
```
This will train a 2D model on the training data and save the model weights to a file (defaults to `./checkpoints/*.pth`), along with logging loss and validation metrics, and sample predictions, viewable in TensorBoard. To view the TensorBoard logs, and monitor training progress, run the following command (assuming you are using the default log directory):

```bash
tensorboard --logdir=./tensorboard
```

You can also train a 3D model by running the same command with `train_3d.py`:

```bash
python train_3d.py
```

For more information on the available options, see the `README.md` in the `examples` folder, as well as the documentation in `examples/train_2d.py` and `examples/train_3d.py`.

## Predict on test data

Example scripts for predicting on test data are provided in the `examples` directory. The scripts are named `predict_2D.py` and `predict_3D.py`, respectively, and are annotated for clarity. You can run one such script by simply running the following on the command line:

```bash
python predict_2D.py
```

See the `README.md` in the `examples` folder for more information on the available options, as well as the documentation in `examples/predict_2D.py` and `examples/predict_3D.py`.

## Post-process model predictions

After making predictions on the test data, you may want to post-process the predictions to improve the results. An example script for post-processing is provided in the `examples` directory, named `postprocess.py` and is annotated for clarity. 

... #TODO: Add more information on post-processing
# TODO: Add post-processing to CLI

## Submit your final predictions

To submit your predictions, first make sure that they are in the correct format (see below), then submit them through [the online submission portal](https://staging.cellmapchallenge.2i2c.cloud/upload). You will need to sign in with your GitHub account to submit your predictions.

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
