Visualization
=============

Purpose -------
 The purpose of this documentation is to provide instructions on how to visualize data and predictions using the `visualize.py` module and the `csc visualize` command. This module allows you to view the raw (EM), groundtruth, predictions, post-processed, and packaged data side by side.

 Usage
 -----
 To visualize the data, you can use the `csc visualize` command. This command serves the data on a local server and opens a browser window with the neuroglancer viewer.

 This command provides multiple options for filtering what data is loaded for viewing. Datasets can be selected with the `-d` option followed by the name of the dataset or a comma-separated list of multiple datasets. A separate viewing window will always be created for each dataset visualized and raw (EM) data will always be shown. If no dataset(s) are specified, all datasets matching any other filters will be shown.
 Similarly, specific crops can also be selected for viewing by using the `-c` option followed by the number of the crop, a comma-separated list of crop numbers (e.g. `-c 234,113`), or "test". `... -c test` will only load the challenge's test crops for viewing. If no crop(s) are specified, all crops matching any other filters will be shown.
 Users can filter what label classes are loaded with the `-C` option followed by the name of the class or a comma-separated list of class names (e.g. `-C nuc,mito,er`). If no class(es) are specified, all available will be shown.
 Which type(s) of data should be visualized can be selected with the `-k` option followed by a string or comma-separated list of strings specifying the data to load. Valid strings include "gt" for groundtruth, "predictions" for predictions, "processed" for post-processed data, and/or "submission" for the data packaged for submission (after resampling). Again, raw (EM) data is always shown. If no type(s) are specified, all available will be shown.

 Examples
 --------
 - Visualize all datasets and crops:
   ```
   csc visualize
   ```

 - Visualize specific datasets, crops, classes, and kinds:
   ```
   csc visualize -d jrc_cos7-1a -c 234,236 -C nuc,cell,mito,er -k gt,predictions
   ```

 Functions
 ---------
 - `visualize(datasets, crops, classes, kinds)`: Visualize datasets and crops in Neuroglancer.
   - Parameters:
     - `datasets` (str | Sequence[str]): The name of the dataset to visualize. Can be a string or a list of strings. Default is "*". If "*", all datasets will be visualized, each in its own browser window/neuroglancer session.
     - `crops` (int | Sequence[int]): The crop number(s) to visualize. Can be an integer, a list of integers, "test", or None. If "test", only the test crops for the competition will be loaded. Default is None. If None, all crops will be visualized.
     - `classes` (str | Sequence[str]): The label class to visualize. Can be a string or a list of strings. Default is "*". If "*", all classes will be visualized.
     - `kinds` (Sequence[str]): The type of layers to visualize. Can be "gt" for groundtruth, "predictions" for predictions, "processed" for post-processed data, and/or "submission" for the data packaged for submission (after resampling). Raw (EM) data is always shown. Default is ["gt", "predictions", "processed", "submission"].

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
  ```
  csc visualize
  ```

- Visualize specific datasets and crops:
  ```
  csc visualize -d jrc_cos7-1a -c 234,236 -C nuc,cell,mito,er -k gt,predictions
  ```

For more detailed information, refer to the [visualization documentation](../docs/source/visualization.rst).
