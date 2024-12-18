Visualization
=============

Purpose
-------
The purpose of this documentation is to provide instructions on how to visualize data and predictions using the `visualize.py` module and the `csc visualize` command.

Visualize.py Module
-------------------
The `visualize.py` module provides functions to visualize data and predictions using neuroglancer. This module allows you to view the raw (EM), groundtruth, predictions, post-processed, and packaged data side by side.

Functions
---------
- `visualize(datasets, crops, classes, kinds)`: Visualize datasets and crops in Neuroglancer.
  - Parameters:
    - `datasets` (str | Sequence[str]): The name of the dataset to visualize. Can be a string or a list of strings. Default is "*". If "*", all datasets will be visualized.
    - `crops` (int | Sequence[int]): The crop number(s) to visualize. Can be an integer or a list of integers, or None. Default is None. If None, all crops will be visualized.
    - `classes` (str | Sequence[str]): The class to visualize. Can be a string or a list of strings. Default is "*". If "*", all classes will be visualized.
    - `kinds` (Sequence[str]): The type of layers to visualize. Can be "gt" for groundtruth, "predictions" for predictions, or "processed" for processed data. Default is ["gt", "predictions", "processed", "submission"].

- `add_layers(viewer, kind, dataset_name, crops, classes)`: Add layers to a Neuroglancer viewer.
  - Parameters:
    - `viewer` (neuroglancer.Viewer): The viewer to add layers to.
    - `kind` (str): The type of layers to add. Can be "gt" for groundtruth, "predictions" for predictions, or "processed" for processed data.
    - `dataset_name` (str): The name of the dataset to add layers for.
    - `crops` (Sequence): The crops to add layers for.
    - `classes` (Sequence[str]): The class(es) to add layers for.

- `get_layer(data_path, layer_type, multiscale)`: Get a Neuroglancer layer from a zarr data path for a LocalVolume.
  - Parameters:
    - `data_path` (str): The path to the zarr data.
    - `layer_type` (str): The type of layer to get. Can be "image" or "segmentation". Default is "image".
    - `multiscale` (bool): Whether the metadata is OME-NGFF multiscale. Default is True.

Usage
-----
To visualize the data and predictions, you can use the `csc visualize` command. This command serves the data and predictions on a local server and opens a browser window with the neuroglancer viewer.

Examples
--------
- Visualize all datasets and crops:
  ```bash
  csc visualize
  ```

- Visualize specific datasets and crops:
  ```bash
  csc visualize -d jrc_cos7-1a -c 234,236 -C nuc,cell,mito,er -k gt,predictions
  ```

For more detailed information, refer to the [visualization documentation](../docs/source/visualization.rst).
