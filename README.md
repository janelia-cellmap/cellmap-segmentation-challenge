<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# CellMap Segmentation Challenge

Welcome to the [**CellMap Segmentation Challenge**](https://janelia.figshare.com/articles/online_resource/CellMap_Segmentation_Challenge/28034561/1?file=51215543) toolbox!

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
8. [Issues](#issues)
7. [Acknowledgements](#acknowledgements)

# Getting started

## Set up your environment

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
Default option will download the data as **Zarr** files. You can download compressed **Zip** files by adding the `--zip` flag. While zip files are faster to download, they can take a long time to unzip. Available Zip files are listed [here](src/cellmap_segmentation_challenge/utils/zip_manifest.csv).

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

### Data Format

For more detailed information on the expected format of the data submitted, refer to the [submission_data_format.rst](docs/source/submission_data_format.rst) file.

# Issues
If you encounter any code-related issues, please open an issue on the [GitHub repository](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/issues) so we can address it as soon as possible. To help us resolve the issue, please provide as much information as possible, including the command you ran, the error message you received, and any other relevant information.

We also have opened [the discussion feature on this repository](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions), so feel free to ask questions or share your thoughts, ideas, progress, etc. there! We are happy to help you with any questions you may have.

# Acknowledgements
[@rhoadesScholar](https://www.linkedin.com/in/rhoadesscholar/) served as the lead developer for this toolbox, supported by contributions from [@aemmav](https://www.linkedin.com/in/emma-avetissian-362089297/), [@d-v-b](https://www.linkedin.com/in/davis-bennett-62922126a/), and [@mzouink](https://www.linkedin.com/in/zouinkhi/). Additional support was provided by [@jonesa3](https://www.linkedin.com/in/alysonpetruncio/), [@rinva](https://www.linkedin.com/in/rebecca-vorimo-831271a0/), [@avweigel](https://www.linkedin.com/in/aubrey-weigel/), and [@yuriyzubov](https://www.linkedin.com/in/yuriizubov/). The CellMap Segmentation Challenge is organized by Janelia's [CellMap Project Team](https://www.janelia.org/project-team/cellmap).
