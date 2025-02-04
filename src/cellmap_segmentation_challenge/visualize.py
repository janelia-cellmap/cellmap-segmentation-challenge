import webbrowser
from glob import glob
from typing import Sequence

import neuroglancer
import numpy as np
import zarr
from upath import UPath

from cellmap_flow.utils.scale_pyramid import ScalePyramid
from cellmap_flow.utils.ds import open_ds_tensorstore

from .config import (
    CROP_NAME,
    PREDICTIONS_PATH,
    PROCESSED_PATH,
    SEARCH_PATH,
    SUBMISSION_PATH,
)
from .utils import TEST_CROPS
from .utils.datasplit import get_dataset_name, get_formatted_fields, get_raw_path

SEARCH_PATHS = {
    "gt": SEARCH_PATH.format(dataset="{dataset}", name=CROP_NAME),
    "predictions": (UPath(PREDICTIONS_PATH) / "{label}").path,
    "processed": (UPath(PROCESSED_PATH) / "{label}").path,
    # "submission": (UPath(SUBMISSION_PATH) / "{crop}" / "{label}").path,
}


def visualize(
    datasets: str | Sequence[str] = "*",
    crops: int | list = ["*"],
    classes: str | Sequence[str] = "*",
    kinds: Sequence[str] = list(SEARCH_PATHS.keys()),
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
        The type of layers to visualize. Can be "gt" for groundtruth, "predictions" for predictions, or "processed" for processed data. Default is ["gt", "predictions", "processed", "submission"].
    """

    # Get all named crops that can be found
    if isinstance(crops, (int, str)):
        crops = [crops]
    if len(crops) == 1 and crops[0] == "test":
        force_em = True
        crops = [crop.id for crop in TEST_CROPS]
        datasets = list(set([crop.dataset for crop in TEST_CROPS]))
    else:
        force_em = False
    for i, crop in enumerate(crops):
        if isinstance(crop, int) or crop.isnumeric():
            crops[i] = f"crop{crop}"

    # Get all named datasets that can be found
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

    # print(f"Found datasets: {dataset_paths}")

    viewer_dict = {}
    for dataset_path in dataset_paths:
        dataset_name = get_dataset_name(dataset_path)

        # Create a new viewer
        viewer = neuroglancer.Viewer()

        # Add the raw dataset
        with viewer.txn() as s:
            # print(f"Found raw data at {get_raw_path(dataset_path)}...")
            s.layers["fibsem"] = get_layer(get_raw_path(dataset_path), "image")

        if force_em:
            viewer_dict[dataset_name] = viewer

        for kind in kinds:
            temp = add_layers(
                viewer,
                kind,
                dataset_name,
                crops,
                classes,
                search_paths=SEARCH_PATHS,
            )

            if temp is not None:
                viewer_dict[dataset_name] = viewer = temp
                print(f"Added {kind} layers for {dataset_name}...")
                webbrowser.open(viewer.get_viewer_url())

    # Output the viewers URL to open in a browser
    for dataset, viewer in viewer_dict.items():
        print(f"{dataset} viewer running at:\n\t{viewer}")

    input("Press Enter to close the viewers...")


def add_layers(
    viewer: neuroglancer.Viewer,
    kind: str,
    dataset_name: str,
    crops: Sequence,
    classes: Sequence[str],
    search_paths=SEARCH_PATHS,
    visible: bool = False,
) -> neuroglancer.Viewer | None:
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
    search_paths : dict, optional
        The search paths to use for finding the data. Default is SEARCH_PATHS.
    visible : bool, optional
        Whether the layers should be visible. Default is False.
    """
    if kind not in search_paths:
        raise ValueError(f"Type must be one of {list(search_paths.keys())}")

    crop_paths = []
    for crop in crops:
        for label in classes:
            if kind == "submission":
                search_path = search_paths[kind].format(
                    crop=crop,
                    label=label,
                )
            else:
                search_path = search_paths[kind].format(
                    dataset=dataset_name,
                    crop=crop,
                    label=label,
                )
            crop_paths.extend(glob(search_path))

    for crop_path in crop_paths:
        formatted_fields = get_formatted_fields(
            crop_path,
            search_paths[kind],
            fields=["{dataset}", "{crop}", "{label}"],
        )

        layer_name = f"{formatted_fields['crop']}/{kind}/{formatted_fields['label']}"
        layer_type = "segmentation" if kind == "gt" else "image"
        with viewer.txn() as s:
            s.layers[layer_name] = get_layer(
                crop_path, layer_type, multiscale=kind != "submission"
            )
            s.layers[layer_name].visible = visible

    if len(crop_paths) == 0:
        return None
    return viewer


def get_layer(
    data_path: str,
    layer_type: str = "image",
    multiscale: bool = True,
) -> neuroglancer.Layer:
    """
    Get a Neuroglancer layer from a zarr data path for a LocalVolume.

    Parameters
    ----------
    data_path : str
        The path to the zarr data.
    layer_type : str
        The type of layer to get. Can be "image" or "segmentation". Default is "image".
    multiscale : bool
        Whether the metadata is OME-NGFF multiscale. Default is True.

    Returns
    -------
    neuroglancer.Layer
        The Neuroglancer layer.
    """
    # Construct an xarray with Tensorstore backend
    # Get metadata
    if multiscale:
        # Add all scales
        layers = []
        scales, metadata = parse_multiscale_metadata(data_path)
        for scale in scales:
            this_path = (UPath(data_path) / scale).path
            image = open_ds_tensorstore(this_path)
            # image = get_image(this_path)

            layers.append(
                neuroglancer.LocalVolume(
                    data=image,
                    dimensions=neuroglancer.CoordinateSpace(
                        scales=metadata[scale]["voxel_size"],
                        units=metadata[scale]["units"],
                        names=metadata[scale]["names"],
                    ),
                    voxel_offset=metadata[scale]["voxel_offset"],
                )
            )
        volume = ScalePyramid(layers)

    else:
        # Handle single scale zarr files
        names = ["z", "y", "x"]
        units = ["nm", "nm", "nm"]
        attrs = zarr.open(data_path, mode="r").attrs.asdict()
        if "voxel_size" in attrs:
            voxel_size = attrs["voxel_size"]
        elif "resolution" in attrs:
            voxel_size = attrs["resolution"]
        elif "scale" in attrs:
            voxel_size = attrs["scale"]
        else:
            voxel_size = [1, 1, 1]

        if "translation" in attrs:
            translation = attrs["translation"]
        elif "offset" in attrs:
            translation = attrs["offset"]
        else:
            translation = [0, 0, 0]

        voxel_offset = np.array(translation) / np.array(voxel_size)

        image = open_ds_tensorstore(data_path)
        # image = get_image(data_path)

        volume = neuroglancer.LocalVolume(
            data=image,
            dimensions=neuroglancer.CoordinateSpace(
                scales=voxel_size,
                units=units,
                names=names,
            ),
            voxel_offset=voxel_offset,
        )

    if layer_type == "segmentation":
        return neuroglancer.SegmentationLayer(source=volume)
    else:
        return neuroglancer.ImageLayer(source=volume)


def get_image(data_path: str):
    import xarray_tensorstore as xt
    import tensorstore

    try:
        return open_ds_tensorstore(data_path)
    except ValueError as e:
        spec = xt._zarr_spec_from_path(data_path)
        array_future = tensorstore.open(spec, read=True, write=False)
        try:
            array = array_future.result()
        except ValueError as e:
            Warning(e)
            UserWarning("Falling back to zarr3 driver")
            spec["driver"] = "zarr3"
            array_future = tensorstore.open(spec, read=True, write=False)
            array = array_future.result()
        return array


def parse_multiscale_metadata(data_path: str):
    metadata = zarr.open(data_path, mode="r").attrs.asdict()["multiscales"][0]
    scales = []
    parsed = {}
    for ds in metadata["datasets"]:
        scales.append(ds["path"])

        names = []
        units = []
        translation = []
        voxel_size = []

        for axis in metadata["axes"]:
            if axis["name"] == "c":
                names.append("c^")
                voxel_size.append(1)
                translation.append(0)
                units.append("")
            else:
                names.append(axis["name"])
                units.append("nm")

        for transform in ds["coordinateTransformations"]:
            if transform["type"] == "scale":
                voxel_size.extend(transform["scale"])
            elif transform["type"] == "translation":
                translation.extend(transform["translation"])

        parsed[ds["path"]] = {
            "names": names,
            "units": units,
            "voxel_size": voxel_size,
            "translation": translation,
            "voxel_offset": np.array(translation) / np.array(voxel_size),
        }
    scales.sort(key=lambda x: int(x[1:]))
    return scales, parsed
