<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# CellMap Segmentation Challenge
Repository of scripts to facilitate participation in CellMap's segmentation challenge. This includes downloading data, simple setups for training 2D and 3D models, workflows for prediction and post-processing on out-of-memory arrays, and evaluation of results against validation data.

# Getting started

## Set up your environment

We recommend using micromamba to create a new environment with the required dependencies. You can install micromamba by following the instructions [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install). The fastest way to do this install is to run the following in your terminal:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Once you have micromamba installed, you can create a new environment with the required dependencies by running the following commands:

```bash
# Create a new environment
micromamba create -n cellmap-segmentation-challenge python=3.10 -y

# Activate the environment
micromamba activate cellmap-segmentation-challenge
```

## Clone the repository

You can clone and install the repository by running the following command:

```bash
# Clone the repository
git clone github.com/janelia-cellmap/cellmap-segmentation-challenge

# Install the repo in editable mode
cd cellmap-segmentation-challenge
pip install -e .
```

## Repository structure

The repository is structured as follows:

...


## Download the data

Once you have installed this package, you can download the challenge data by running the following command:

```bash
csc fetch-data --crops all --dest path/to/dest
```

This will retrieve groundtruth data and corresponding EM data for each crop and save it at a path 
of your choosing on your local filesystem.

To see all the options for the the `fetch-data` command, run 

```bash
csc fetch-data --help
```

## Train a model

You can train a model by running the following command:

```bash
...
```

## Predict on test data

You can predict on test data by running the following command:

```bash
...
```

## Evaluate your predictions

You can evaluate your predictions by running the following command:

```bash
...
```
