<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# CellMap Segmentation Challenge

Welcome to the **CellMap Segmentation Challenge** toolbox!

This Python package provides a simple and flexible API for:

- [Accessing challenge data](#download-the-data)
- [Running model training](#train-a-model)
- [Making predictions](#predict-on-test-data)
- [Evaluating results](#post-process-model-predictions)

The package is built on top of the [`cellmap-data`](https://github.com/janelia-cellmap/cellmap-data) package, which offers tools for interacting with the CellMap data format. Whether you're a beginner or advanced user, this package is designed to be easy to use and highly extensible.

## Table of Contents
1. [Getting Started](#getting-started)
    - [Set up your environment](#set-up-your-environment)
    - [Clone the repository](#clone-the-repository)
    - [Repository structure](#repository-structure)
    - [Download the data](#download-the-data)
2. [Train a model](#train-a-model)
3. [Predict on test data](#predict-on-test-data)
4. [Post-process model predictions](#post-process-model-predictions)
5. [Visualize data and predictions](#visualize-data-and-predictions)
6. [Submit your final predictions](#submit-your-final-predictions)
    - [Data format](#data-format)
7. [Acknowledgements](#acknowledgements)

# Getting started

We provide two options for installation of the CellMap Segmentation Challenge software.
1. Using `pixi`
2. Using `micromamba` and `pip`

`pixi` is a new package management tool that can handle both conda and PyPI packages. This is particularly useful installation on Windows or advanced setups.

For basic usage, we also provide a `pip` based installation method using micromamba to create a virtual environment.
You can also [use pyvenv to create a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-a-new-virtual-environment).

## Getting Started with Pixi

First, install `pixi` by visiting [pixi.sh](https://pixi.sh/latest/#installation) or using the instructions below.

### Linux & macOS

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Windows

#### Powershell
```
iwr -useb https://pixi.sh/install.ps1 | iex
```

#### winget

```
winget install prefix-dev.pixi
```

### Clone the repository

You can clone and install the repository by running the following command:

```bash
# Clone the repository
git clone https://github.com/janelia-cellmap/cellmap-segmentation-challenge

# Install the repo in editable mode
cd cellmap-segmentation-challenge
pixi install
```

### Using the pixi environment

To use the pixi environment, you can either prefix the commands below with `pixi run` (e.g. `pixi run csc --fetch-data`),
or use `pixi shell` which is similar to activating an environment with `conda` or `mamba`.

## Getting started with micromamba and pip

### Set up your environment

We recommend using micromamba to create a new environment with the required dependencies. You can install micromamba by following the instructions [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install). The fastest way to do this install, if you are using Linux, macOS, or Git Bash on Windows, is to run the following in your terminal:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

If you are using Windows Powershell (not recommended), you should be able to install micromamba by running the following command (NOTE: you will need to replace `micromamba` with `Invoke-Mamba` in the commands below):

```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)
```

Once you have micromamba installed, you can create a new environment with the required dependencies by running the following commands:

```bash
# Create a new environment
micromamba create -n csc -c conda-forge -y python==3.11

# Activate the environment
micromamba activate csc
```

### Clone the repository

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
│   ├── train_2D.py
│   ├── train_3D.py
│   ├── predict_2D.py
│   ├── predict_3D.py
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
csc fetch-data
```

This will retrieve all of the groundtruth data and corresponding EM data for each crop and save it to `./data` relative to this `README.md` on your local filesystem. This is the default `BASE_DATA_PATH` defined in the global configuration file (`config.py`). You can change the configured paths with this file as you like.

Additionally, you can request raw data in all resolutions (not just those matching the annotations), extra raw data beyond the borders of the annotation crops (i.e. padding), custom download location, and more. To see all the options for the `fetch-data` command, run 

```bash
csc fetch-data --help
```

Downloading the data may take some time, depending on your internet connection and the size of the data based on your download preferences. For reference, the download time for the default options on a MacBook Pro Apple M3 Max with 128 GB of memory through WiFi was 1.6 hours, and required approximately 36.7 GB of storage. Downloading full-resolution data for all crops with 128 voxels of padding took 6.6 hours on an internet connection with 820Mbps download speed, and required approximately 1.18TB of storage.

# Train a model

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

Once you are satisfied with your model, you can use it to make predictions, as discussed in the next section.

# Predict on test data

Example scripts for predicting on test data are also provided in the `examples` directory. The scripts are named `predict_2D.py` and `predict_3D.py`, respectively. You can use one such script by simply running the following on the command line:

```bash
python predict_2D.py
```

There is also a built in command to run predictions, given the path to a model (or training) configuration file. To learn more, you can run the following in the terminal:

```bash
csc predict --help
```

Also see the `README.md` in the `examples` folder for more information on the available options, as well as the documentation in `examples/predict_2D.py` and `examples/predict_3D.py`.

# Post-process model predictions

After making predictions on the test data, you may want to post-process the predictions to improve the results. Example scripts for post-processing are provided in the `examples` directory, named `process_2D.py` and `process_3D.py`. They are annotated for clarity. You can run one such script by running the following on the command line:

```bash
csc process process_2D.py
```

This functionality allows you to define any `process_func` that takes in a batch of predictions and returns a batch of post-processed predictions. This can be used to apply any post-processing steps you like, such as thresholding, merging IDs for connected components, filtering based on object size, etc..

For more information on the available options and how post-processing works, see the `README.md` in the `examples` folder, as well as the documentation in `examples/process_2D.py` and `examples/process_3D.py`, or run the following in the terminal:

```bash
csc process --help
```

During evaluations of submissions, for instance segmentation evaluated classes, connected components are computed on the supplied masks and the resulting instance IDs are assigned to each connected component. This will not merge already uniquely IDed objects. Thus, you do not need to run connected components on before submission, but you may wish to execute more advanced post-processing for instance segmentation, such as watershed.

# Visualize data and predictions

To visualize data and predictions, you can view them with neuroglancer. As an example, the following command will visualize the groundtruth and predictions for the jrc_cos7-1a dataset, crops 234 and 236, and label classes nuc, cell, mito, and er:

```bash
csc visualize -d jrc_cos7-1a -c 234,236 -C nuc,cell,mito,er -k gt,predictions
```

This will serve the data and predictions on a local server, and open a browser window with the neuroglancer viewer. You can then navigate through the data and predictions, and compare them side by side. The default call with no arguments will do this for all datasets and groundtruth, prediction, and processed crops for all label classes found at the search paths defined in the global configuration file (`config.py`). You can also specify particular datasets, crops, labels classes, and whether to show groundtruth, predictions, or processed data to visualize. 

To visualize all of the data and predictions, you can run the following command:

```bash
csc visualize
```

Run the following command to see all the options:

```bash
csc visualize --help
```

# Submit your final predictions

Once submissions open, ensure your predictions are in the correct format (see below), then upload them through [the online submission portal](https://cellmapchallenge.janelia.org/submissions/). You will need to sign in with your GitHub account to submit your predictions.

For convenience, if you have followed the prediction and processing steps described above and in the example scripts, you can use the following command to zip your predictions in the correct format:

```bash
csc pack-results
```
To see all the options for the `pack-results` command, see the README in the `examples` folder, or run 

```bash
csc pack-results --help
```

### Data format

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

# Acknowledgements
[@rhoadesScholar](https://www.linkedin.com/in/rhoadesscholar/) served as the lead developer for this toolbox, supported by contributions from [@aemmav](https://www.linkedin.com/in/emma-avetissian-362089297/), [@d-v-b](https://www.linkedin.com/in/davis-bennett-62922126a/), and [@mzouink](https://www.linkedin.com/in/zouinkhi/). Additional support was provided by [@jonesa3](https://www.linkedin.com/in/alysonpetruncio/), [@rinva](https://www.linkedin.com/in/rebecca-vorimo-831271a0/), [@avweigel](https://www.linkedin.com/in/aubrey-weigel/), and [@yuriyzubov](https://www.linkedin.com/in/yuriizubov/). The CellMap Segmentation Challenge is organized by Janelia's [CellMap Project Team](https://www.janelia.org/project-team/cellmap).