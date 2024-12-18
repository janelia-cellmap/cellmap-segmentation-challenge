# CellMap Segmentation Challenge Models

This directory contains the models used in the CellMap Segmentation Challenge. The models are implemented in PyTorch and currently include:

- UNet_2D
- UNet_3D
- ResNet
- ViTVNet

## Model Loading

The `model_load.py` file contains functions for loading model checkpoints:

- `load_latest(search_path, model)`: Load the latest checkpoint from a directory into a model.
- `load_best_val(logs_save_path, model_save_path, model, low_is_best=True)`: Load the model weights with the best validation score, as recorded in tensorboard logs, from a directory into an existing model object.

For more detailed information about the models and their usage, refer to the [documentation files](../../docs/source/load_model_weights.rst).

 ## Training Configuration

 The following parts can be included in the training configuration file:

 - `model_name`: Name of the model to use. If the config file constructs the PyTorch model, this name can be anything. If the config file does not construct the PyTorch model, the model_name will need to specify which included architecture to use. This includes '2d_unet', '2d_resnet', '3d_unet', '3d_resnet', and 'vitnet'. Default is '2d_unet'.
 - `model_kwargs`: Dictionary of keyword arguments to pass to the model constructor (specified by `model_name`). Default is {}. If the PyTorch `model` is passed, this will be ignored.

 ## Links to Relevant Documentation

 For more detailed information about the models and their usage, refer to the following documentation files:

 - [Main README.md](../../README.md)
 - [Training Documentation](../../docs/source/training.rst)
