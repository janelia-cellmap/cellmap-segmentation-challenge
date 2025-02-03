import shutil
import sys
from time import time
import numpy as np
import requests
from tqdm import tqdm
from cellmap_segmentation_challenge.utils import get_tested_classes
from cellmap_segmentation_challenge import TRUTH_PATH
import zarr
from skimage.measure import label as relabel

from upath import UPath


def format_coordinates(coordinates):
    """
    Format the coordinates to a string.

    Parameters
    ----------
    coordinates : list
        List of coordinates.

    Returns
    -------
    str
        Formatted string.
    """
    return f"[{';'.join([str(c) for c in coordinates])}]"


def construct_test_crop_manifest(
    path_root: str,
    search_path: str = "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}",
    write_path: str | None = (UPath(__file__).parent / "test_crop_manifest.csv").path,
    verbose: bool = False,
) -> None | list[str]:
    """
    Construct a manifest file for testing crops from a given path.

    Parameters
    ----------
    path_root : str
        Path to the directory containing the datasets.
    search_path : str, optional
        Format string to search for the crops. The default is "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}". The function assumes that the keys appear in the file tree in the following order: 1) "path_root", 2) "dataset", 3) "crop", 4) "label"
    write_path : str, optional
        Path to write the manifest file. The default is "test_crop_manifest.csv".
    verbose : bool, optional
        Print verbose output. The default is False.
    """
    # Get the tested classes
    tested_classes = set(get_tested_classes())

    # Construct the manifest
    manifest = [
        "crop_name,dataset,class_label,voxel_size,translation,shape",
    ]

    # Get datasets
    datasets = [
        d.name
        for d in UPath(
            search_path.split("{dataset}")[0].format(path_root=path_root)
        ).iterdir()
        if d.is_dir()
    ]

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Get crops
        crops = [
            d.name
            for d in UPath(
                search_path.split("{crop}")[0].format(
                    path_root=path_root, dataset=dataset
                )
            ).iterdir()
            if d.is_dir()
        ]
        for crop in crops:
            if verbose:
                print(f"\tProcessing crop: {crop}")

            # Get labels in crop
            had_classes = set(
                [
                    d.name
                    for d in UPath(
                        search_path.split("{label}")[0].format(
                            path_root=path_root, dataset=dataset, crop=crop
                        )
                    ).iterdir()
                    if d.is_dir()
                ]
            )

            # Filter for tested classes
            labels = list(had_classes.intersection(tested_classes))

            for label in labels:
                if verbose:
                    print(f"\t\tProcessing label: {label}")
                # Get the zarr file
                zarr_file = zarr.open(
                    search_path.format(
                        path_root=path_root, dataset=dataset, crop=crop, label=label
                    ),
                    mode="r",
                )

                # Get the metadata
                metadata = zarr_file.attrs.asdict()["multiscales"][0]["datasets"][0][
                    "coordinateTransformations"
                ]
                for meta in metadata:
                    if meta["type"] == "translation":
                        translation = format_coordinates(meta["translation"])
                    elif meta["type"] == "scale":
                        voxel_size = format_coordinates(meta["scale"])
                shape = format_coordinates(zarr_file["s0"].shape)
                manifest.append(
                    f"{crop.replace('crop', '')},{dataset},{label},{voxel_size},{translation},{shape}"
                )
                if verbose:
                    print(
                        f"\t\t\tScale: {voxel_size}\n\t\t\tTranslation: {translation}\n\t\t\tShape: {shape}"
                    )
    if write_path is None:
        return manifest

    # Write the manifest
    with open(write_path, "w") as f:
        f.write("\n".join(manifest))

    print(f"Manifest written to: {write_path}")


def construct_truth_dataset(
    path_root: str,
    search_path: str = "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}",
    destination: str = TRUTH_PATH,
    write_path: str = "{crop}/{label}",
):
    """
    Construct a consolidated Zarr file for the groundtruth datasets, to use for evaluation.

    Parameters
    ----------
    path_root : str
        Path to the directory containing the datasets.
    search_path : str, optional
        Format string to search for the crops. The default is "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}". The function assumes that the keys appear in the file tree in the following order: 1) "path_root", 2) "dataset", 3) "crop", 4) "label"
    destination : str, optional
        Path to write the consolidated Zarr file. The default is "cellmap-segmentation-challenge/data/ground_truth.zarr".
    write_path : str, optional
        Format string to write the crops to within the destination Zarr. The default is "{crop}/{label}".
    """
    start_time = time()

    # Get the test crop manifested
    manifest = construct_test_crop_manifest(path_root, search_path, write_path=None)

    # Open the destination Zarr folder
    if UPath(destination).exists():
        print(f"Removing existing ground truth dataset at: {destination}")
        shutil.rmtree(destination)
    # ground_truth = zarr.open_group(destination, mode="w")
    ground_truth = zarr.open_group(destination, mode="a")

    # Make a pool for parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pool = ThreadPoolExecutor()

    # Copy the ground truth datasets
    futures = []
    crops_started = set()
    for line in tqdm(manifest[1:], desc="Formatting ground truth..."):
        crop = line.split(",")[0]
        if crop not in crops_started:
            crops_started.add(crop)
            ground_truth.create_group(f"crop{crop}")
        futures.append(
            pool.submit(copy_gt, line, search_path, path_root, write_path, ground_truth)
        )

    for future in tqdm(as_completed(futures), total=len(futures), desc="Copying..."):
        future.result()

    print(f"Ground truth dataset written to: {destination}")
    print(f"Done in {time() - start_time}!")


def copy_gt(line, search_path, path_root, write_path, ground_truth):
    # Get the metadata from the manifest
    crop, dataset, class_label, voxel_size, translation, shape = line.split(",")
    crop_name = f"crop{crop}"
    voxel_size = eval(voxel_size.replace(";", ","))
    translation = eval(translation.replace(";", ","))
    shape = eval(shape.replace(";", ","))

    # Open the source ground truth zarr file
    path = search_path.format(
        path_root=path_root, dataset=dataset, crop=crop_name, label=class_label
    )
    zarr_file = zarr.open(path, mode="r")

    # Write the dataset to the destination Zarr
    print(f"Writing {write_path.format(crop=crop_name, label=class_label)}")
    dataset = ground_truth.create_dataset(
        write_path.format(crop=crop_name, label=class_label),
        data=zarr_file["s0"],
        shape=shape,
        dtype=zarr_file["s0"].dtype,
        overwrite=True,
        # fill_value=0,
        dimension_separator="/",
    )
    dataset.attrs["voxel_size"] = voxel_size
    dataset.attrs["translation"] = translation
    dataset.attrs["shape"] = shape


# %%


# Helper functions for simulating predictions
def simulate_predictions_iou(true_labels, iou):
    # TODO: Add false positives (only makes false negatives currently)

    pred_labels = np.zeros_like(true_labels)
    for i in np.unique(true_labels):
        if i == 0:
            continue
        pred_labels[true_labels == i] = np.random.choice(
            [i, 0], np.sum(true_labels == i), p=[iou, 1 - iou]
        )

    pred_labels = relabel(pred_labels, connectivity=len(pred_labels.shape))
    return pred_labels


def simulate_predictions_accuracy(true_labels, accuracy):
    shape = true_labels.shape
    true_labels = true_labels.flatten()

    # Get the total number of labels
    n = len(true_labels)

    # Calculate the number of correct predictions
    num_correct = int(accuracy * n)

    # Create an array to store the simulated predictions (copy the true labels initially)
    simulated_predictions = np.copy(true_labels)

    # Randomly select indices to be incorrect
    incorrect_indices = np.random.choice(n, size=n - num_correct, replace=False)

    # Flip the labels at the incorrect indices
    for idx in incorrect_indices:
        # Assuming binary classification (0 or 1), flip the label
        simulated_predictions[idx] = 1 - simulated_predictions[idx]

    # Relabel the predictions
    simulated_predictions = simulated_predictions.reshape(shape)
    simulated_predictions = relabel(simulated_predictions, connectivity=len(shape))

    return simulated_predictions


def download_file(url, dest):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)


def format_string(string: str, format_kwargs: dict) -> str:
    """
    Convenience function to format a string with only the keys present in both the stringand in the `format_kwargs`. When all keys in the `format_kwargs` are present in `string` (in brackets), the function will return `string.format(**format_kwargs)` exactly. When none of the keys in the `format_kwargs` are present in the string, the function will return the original string, without error.

    Parameters
    ----------
    string : str
        The string to format.
    format_kwargs : dict
        The dictionary of key-value pairs to format the string with.

    Returns
    -------
    str
        The formatted string

    Examples
    --------
    format_string("this/{thing}", {})  # returns "this/{thing}"
    format_string("this/{thing}", {"thing":"that", "but":"not this"}) # returns "this/that"
    """
    new_kwargs = {}
    # Find the keys that are present in the string
    for key_chunk in string.split("{")[1:]:
        key = key_chunk.split("}")[0]
        if key in format_kwargs:
            new_kwargs[key] = format_kwargs[key]
        else:
            new_kwargs[key] = "{" + key + "}"
    string = string.format(**new_kwargs)
    return string


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils.py <path_root>")
        sys.exit(1)
    elif len(sys.argv) == 2 or sys.argv[2] == "dataset":
        construct_truth_dataset(
            sys.argv[1],
        )
    elif sys.argv[2] == "manifest":
        construct_test_crop_manifest(sys.argv[1], verbose=True)
