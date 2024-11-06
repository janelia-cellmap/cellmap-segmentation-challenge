from typing import Optional, Sequence
import neuroglancer
from glob import glob
from upath import UPath
from .config import CROP_NAME, REPO_ROOT, SEARCH_PATH, PREDICTIONS_PATH, PROCESSED_PATH
from .utils.datasplit import get_raw_path, get_dataset_name, get_formatted_fields

search_paths = {
    "gt": SEARCH_PATH.format(dataset="{dataset}", name=CROP_NAME),
    "predictions": PREDICTIONS_PATH,
    "processed": PROCESSED_PATH,
}


def visualize(
    datasets: str | Sequence[str] = "*",
    crops: int | Sequence = ["*"],
    classes: str | Sequence[str] = "*",
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

    viewer_dict = {}
    for dataset_path in dataset_paths:
        dataset_name = get_dataset_name(dataset_path)

        # Make the neuroglancer viewer for this dataset
        viewer_dict[dataset_name] = neuroglancer.Viewer()

        # Add the raw dataset
        with viewer_dict[dataset_name].txn() as s:
            s.layers["fibsem"] = neuroglancer.ImageLayer(
                source=get_raw_path(dataset_path),
            )

        # Add every groundtruth layer matching the crops and classes
        crop_paths = []
        for crop in crops:
            for label in classes:
                crop_paths.extend(
                    glob(
                        SEARCH_PATH.format(
                            dataset=dataset_name,
                            name=CROP_NAME.format(crop=crop, label=label),
                        )
                    )
                )

        for crop_path in crop_paths:
            formatted_fields = get_formatted_fields(
                crop_path,
                SEARCH_PATH.format(dataset="{dataset}", name=CROP_NAME),
                fields=["{dataset}", "{crop}", "{label}"],
            )

            layer_name = f"{formatted_fields['crop']}/{formatted_fields['label']}"
            with viewer_dict[dataset_name].txn() as s:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=crop_path,
                )

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
        layer_class = (
            neuroglancer.ImageLayer
            if kind == "predictions"
            else neuroglancer.SegmentationLayer
        )
        with viewer.txn() as s:
            s.layers[layer_name] = layer_class(source=crop_path)
