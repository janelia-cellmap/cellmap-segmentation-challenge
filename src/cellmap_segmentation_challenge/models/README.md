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

## Links to Relevant Documentation

For more detailed information about the models and their usage, refer to the following documentation files:

- [Main README.md](../../README.md)
- [Training Documentation](../../docs/source/training.rst)
