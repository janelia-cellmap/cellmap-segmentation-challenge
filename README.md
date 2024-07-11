<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# CellMap Segmentation Challenge
Repository of scripts to facilitate participation in CellMap's segmentation challenge. This includes downloading data, simple setups for training 2D and 3D models, workflows for prediction and post-processing on out-of-memory arrays, and evaluation of results against validation data.

# Getting started

## Set up your environment

We recommend using micromamba to create a new environment with the required dependencies. You can install micromamba by following the instructions [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install). Once you have micromamba installed, you can create a new environment with the required dependencies by running the following commands:

```bash
micromamba create -n cellmap-segmentation-challenge python=3.10 pytorch torchvision numpy tqdm cellmap-data -c pytorch -c conda-forge