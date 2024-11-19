import webbrowser
from glob import glob
from typing import Sequence

import neuroglancer
import numpy as np
import tensorstore
import xarray_tensorstore as xt
import zarr
from upath import UPath

from .config import CROP_NAME, PREDICTIONS_PATH, PROCESSED_PATH, SEARCH_PATH
from .evaluate import TEST_CROPS
from .utils.datasplit import get_dataset_name, get_formatted_fields, get_raw_path

search_paths = {
    "gt": SEARCH_PATH.format(dataset="{dataset}", name=CROP_NAME),
    "predictions": (UPath(PREDICTIONS_PATH) / "{label}").path,
    "processed": (UPath(PROCESSED_PATH) / "{label}").path,
}


def visualize(
    datasets: str | Sequence[str] = "*",
    crops: int | list = ["*"],  # TODO: Add "test" crops
    classes: str | Sequence[str] = "*",
    kinds: Sequence[str] = list(search_paths.keys()),
):
    """
    Visualize datasets and crops in Neuroglancer.

    Parameters
    ----------
    datasets : str | Sequence[str], optional
        The name of the dataset to visualize. Can be a string or a list of strings. Default is "*". If "*", all datasets will be visualized.
    crops : int | Sequence[int], optional
        The crop number(s) to visualize. Can be an integer or a list of integers, or None. Default is None. If None, all crops will be visualized.
    classes : str | Sequence[str], optional
        The class to visualize. Can be a string or a list of strings. Default is "*". If "*", all classes will be visualized.
    kinds : Sequence[str], optional
        The type of layers to visualize. Can be "gt" for groundtruth, "predictions" for predictions, or "processed" for processed data. Default is ["gt", "predictions", "processed"].
    """

    # Get all matching datasets that can be found
    if isinstance(datasets, str):
        dataset_paths = glob(SEARCH_PATH.format(dataset=datasets, name=""))
    else:
        dataset_paths = []
        for dataset in datasets:
            found_paths = glob(SEARCH_PATH.format(dataset=dataset, name=""))
            if len(found_paths) == 0:
                print(f"No datasets found for dataset: {dataset}")
            dataset_paths.extend(found_paths)

    if isinstance(classes, str):
        classes = [classes]

    if isinstance(crops, (int, str)):
        crops = [crops]
    if len(crops) == 1 and crops[0] == "test":
        crops = TEST_CROPS
    for i, crop in enumerate(crops):
        if isinstance(crop, int) or crop.isnumeric():
            crops[i] = f"crop{crop}"

    viewer_dict = {}
    for dataset_path in dataset_paths:
        dataset_name = get_dataset_name(dataset_path)

        # Make the neuroglancer viewer for this dataset
        viewer_dict[dataset_name] = neuroglancer.Viewer()

        # Add the raw dataset
        with viewer_dict[dataset_name].txn() as s:
            s.layers["fibsem"] = get_layer(get_raw_path(dataset_path), "image")

        for kind in kinds:
            viewer_dict[dataset_name] = add_layers(
                viewer_dict[dataset_name],
                kind,
                dataset_name,
                crops,
                classes,
            )

    # Output the viewers URL to open in a browser
    for dataset, viewer in viewer_dict.items():
        webbrowser.open(viewer.get_viewer_url())
        print(f"{dataset} viewer running at:\n\t{viewer}")

    input("Press Enter to close the viewers...")


def add_layers(
    viewer: neuroglancer.Viewer,
    kind: str,
    dataset_name: str,
    crops: Sequence,
    classes: Sequence[str],
):
    """
    Add layers to a Neuroglancer viewer.

    Parameters
    ----------
    viewer : neuroglancer.Viewer
        The viewer to add layers to.
    kind : str
        The type of layers to add. Can be "gt" for groundtruth, "predictions" for predictions, or "processed" for processed data.
    dataset_name : str
        The name of the dataset to add layers for.
    crops : Sequence
        The crops to add layers for.
    classes : Sequence[str]
        The class(es) to add layers for.
    """
    if kind not in search_paths:
        raise ValueError(f"Type must be one of {list(search_paths.keys())}")

    crop_paths = []
    for crop in crops:
        for label in classes:
            crop_paths.extend(
                glob(
                    search_paths[kind].format(
                        dataset=dataset_name,
                        crop=crop,
                        label=label,
                    )
                )
            )

    for crop_path in crop_paths:
        formatted_fields = get_formatted_fields(
            crop_path,
            search_paths[kind],
            fields=["{dataset}", "{crop}", "{label}"],
        )

        layer_name = f"{formatted_fields['crop']}/{kind}/{formatted_fields['label']}"
        layer_type = "segmentation" if kind == "gt" else "image"
        with viewer.txn() as s:
            s.layers[layer_name] = get_layer(crop_path, layer_type)

    return viewer


def get_layer(data_path: str, layer_type: str = "image") -> neuroglancer.Layer:
    """
    Get a Neuroglancer layer from a zarr data path for a LocalVolume.

    Parameters
    ----------
    data_path : str
        The path to the zarr data.
    layer_type : str
        The type of layer to get. Can be "image" or "segmentation". Default is "image".

    Returns
    -------
    neuroglancer.Layer
        The Neuroglancer layer.
    """
    # Construct an xarray with Tensorstore backend
    # TODO: Make this work with multiscale properly
    spec = xt._zarr_spec_from_path((UPath(data_path) / "s0").path)
    array_future = tensorstore.open(spec, read=True, write=False)
    try:
        array = array_future.result()
    except ValueError as e:
        Warning(e)
        UserWarning("Falling back to zarr3 driver")
        spec["driver"] = "zarr3"
        array_future = tensorstore.open(spec, read=True, write=False)
        array = array_future.result()

    # Get metadata
    metadata = zarr.open(data_path).attrs.asdict()["multiscales"][0]
    names = []
    units = []
    scales = []
    offset = []
    for axis in metadata["axes"]:
        if axis["name"] == "c":
            names.append("c^")
            scales.append(1)
            offset.append(0)
            units.append("")
        else:
            names.append(axis["name"])
            units.append("nm")

    for ds in metadata["datasets"]:
        if ds["path"] == "s0":
            for transform in ds["coordinateTransformations"]:
                if transform["type"] == "scale":
                    scales.extend(transform["scale"])
                elif transform["type"] == "translation":
                    offset.extend(transform["translation"])
            break

    voxel_offset = np.array(offset) / np.array(scales)
    volume = neuroglancer.LocalVolume(
        data=array,
        dimensions=neuroglancer.CoordinateSpace(
            scales=scales,
            units=units,
            names=names,
        ),
        voxel_offset=voxel_offset,
    )
    if layer_type == "segmentation":
        return neuroglancer.SegmentationLayer(source=volume)
    return neuroglancer.ImageLayer(source=volume)
