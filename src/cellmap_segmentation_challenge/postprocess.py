import os
import numpy as np
from upath import UPath
from tqdm import tqdm
from funlib.persistence import open_ds, prepare_ds
from importlib.machinery import SourceFileLoader
from typing import Optional
from skimage.transform import resize, rescale
import zarr


def threshold_volume(
    input_container: str | UPath,
    threshold: float | list[float] | dict[str, float] = 0.5,
    output_path: str | UPath = "thresholded.zarr",
    labels: Optional[list[str]] = None,
):
    """
    Threshold a volume in a zarr container.

    Parameters
    ----------
    input_container : str | UPath
        The path to the zarr container containing the data for each label.
    threshold : float | list[float] | dict[str, float]
        The threshold(s) to apply to each label. If a float, the same threshold is applied to all labels. If a list, the thresholds are applied to the labels in order. If a dict, the thresholds are applied to the labels with the corresponding keys.
    output_path : UPath
        The path to the zarr container to write the thresholded data to.
    labels : Optional[list[str]], optional
        The labels to threshold in the zarr container. If None, all labels are thresholded. Default is None.
    """
    if labels is None:
        labels = zarr.open_group(input_container).keys()
    for i, label in enumerate(tqdm(labels)):
        if isinstance(threshold, dict):
            if label not in threshold:
                print(f"Skipping {label} as it is not in the threshold dict")
                continue
            threshold_value = threshold[label]
        elif isinstance(threshold, list):
            if len(threshold) <= i:
                continue
            threshold_value = threshold[i]
        else:
            threshold_value = threshold
        input_ds = open_ds(UPath(input_container) / label)

        data = input_ds[:]
        data = data > threshold_value

        output_ds = prepare_ds(
            output_path / label,
            data.shape,
            offset=input_ds.offset,
            voxel_size=input_ds.voxel_size,
        )
        output_ds[:] = data


def process_volume(
    input_container: str | UPath,
    process_func: callable | list[callable] | dict[str, callable] | os.PathLike,
    output_path: str | UPath,
    labels: Optional[list[str]] = None,
):
    """
    Postprocess a volume in a zarr container with an arbitrary function.

    Parameters
    ----------
    input_container : str | UPath
        The path to the zarr container containing the data for each label.
    process_func : callable | list[callable] | dict[str, callable] | os.PathLike
        The function(s) to apply to each label. If a callable, the same function is applied to all labels. If a list, the functions are applied to the labels in order. If a dict, the functions are applied to the labels with the corresponding keys. If an os.PathLike, the function is loaded from the file at the path (the function should be called `process_func` in the python file). This last option should take a numpy array as input and return a numpy array as output. This allows for more complex postprocessing functions to be used.
    output_path : UPath
        The path to the zarr container to write the thresholded data to.
    labels : Optional[list[str]], optional
        The labels to process in the zarr container. If None, all labels are processed. Default is None.
    """
    if labels is None:
        labels = zarr.open_group(input_container).keys()
    if isinstance(process_func, os.PathLike):
        process_func = (
            SourceFileLoader("process_func", process_func).load_module().process_func
        )
    for i, label in enumerate(tqdm(labels)):
        if isinstance(process_func, dict):
            if label not in process_func:
                print(f"Skipping {label} as it is not in the process_func dict")
                continue
            label_process_func = process_func[label]
        elif isinstance(process_func, list):
            if len(process_func) <= i:
                continue
            label_process_func = process_func[i]
        else:
            label_process_func = process_func
        input_ds = open_ds(UPath(input_container) / label)

        data = input_ds[:]
        data = label_process_func(data)

        output_ds = prepare_ds(
            output_path / label,
            data.shape,
            offset=input_ds.offset,
            voxel_size=input_ds.voxel_size,
        )
        output_ds[:] = data


def rescale_volume(
    input_container: str | UPath,
    output_path: str | UPath,
    output_voxel_size: list[float] | list[list[float]] | dict[str, list[float]],
    labels: Optional[list[str]] = None,
):
    """
    Rescale volumes within a zarr container.

    Parameters
    ----------
    input_container : str | UPath
        The path to the zarr container containing the data for each label.
    output_path : UPath
        The path to the zarr container to write the rescaled data to.
    output_voxel_size : list[float] | list[list[float]] | dict[str, list[float]]
        The voxel size(s) to rescale the labels to. If a list, the same voxel size is applied to all labels. If a list of lists, the voxel sizes are applied to the labels in order. If a dict, the voxel sizes are applied to the labels with the corresponding keys.
    labels : Optional[list[str]], optional
        The labels to rescale in the zarr container. If None, all labels are rescaled. Default is None
    """
    if labels is None:
        labels = zarr.open_group(input_container).keys()
    for i, label in enumerate(tqdm(labels)):
        if isinstance(output_voxel_size, dict):
            if label not in output_voxel_size:
                print(f"Skipping {label} as it is not in the output_voxel_size dict")
                continue
            voxel_size = output_voxel_size[label]
        elif isinstance(output_voxel_size, list):
            if len(output_voxel_size) <= i:
                continue
            voxel_size = output_voxel_size[i]
        else:
            voxel_size = output_voxel_size
        input_ds = open_ds(UPath(input_container) / label)
        input_voxel_size = input_ds.voxel_size
        scale = np.array(input_voxel_size) / np.array(voxel_size)
        data = input_ds[:]
        data = rescale(data, scale, order=0)
        output_ds = prepare_ds(
            output_path / label,
            data.shape,
            offset=input_ds.offset,
            voxel_size=voxel_size,
        )
        output_ds[:] = data


def resize_volume(
    input_container: str | UPath,
    output_path: str | UPath,
    output_shape: list[int] | list[list[int]] | dict[str, list[int]],
    labels: Optional[list[str]] = None,
):
    """
    Resize volumes within a zarr container to a given shape.

    Parameters
    ----------
    input_container : str | UPath
        The path to the zarr container containing the data for each label.
    output_path : UPath
        The path to the zarr container to write the resized data to.
    output_shape : list[int] | list[list[int]] | dict[str, list[int]]
        The shape(s) to resize the labels to. If a list, the same shape is applied to all labels. If a list of lists, the shapes are applied to the labels in order. If a dict, the shapes are applied to the labels with the corresponding keys.
    labels : Optional[list[str]], optional
        The labels to resize in the zarr container. If None, all labels are resized. Default is None
    """
    if labels is None:
        labels = zarr.open_group(input_container).keys()
    for i, label in enumerate(tqdm(labels)):
        if isinstance(output_shape, dict):
            if label not in output_shape:
                print(f"Skipping {label} as it is not in the output_shape dict")
                continue
            shape = output_shape[label]
        elif isinstance(output_shape, list):
            if len(output_shape) <= i:
                continue
            shape = output_shape[i]
        else:
            shape = output_shape
        input_ds = open_ds(UPath(input_container) / label)
        data = input_ds[:]
        data = resize(data, shape, order=0)
        output_ds = prepare_ds(
            output_path / label,
            data.shape,
            offset=input_ds.offset,
            voxel_size=input_ds.voxel_size,
        )
        output_ds[:] = data
