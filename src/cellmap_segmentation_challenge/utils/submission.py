import zarr
import zarr.errors
from upath import UPath
import os
import zipfile
import functools
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from cellmap_segmentation_challenge.utils import format_string, TEST_CROPS
from cellmap_segmentation_challenge.evaluate import match_crop_space
from cellmap_segmentation_challenge import SUBMISSION_PATH, PROCESSED_PATH

import logging


def save_numpy_class_labels_to_zarr(
    save_path, test_volume_name, label_name, labels, overwrite=False, attrs=None
):
    """
    Save a single 3D numpy array of class labels to a
    Zarr-2 file with the required structure.

    Args:
        save_path (str): The path to save the Zarr-2 file (ending with <filename>.zarr).
        test_volume_name (str): The name of the test volume.
        label_names (str): The names of the labels.
        labels (np.ndarray): A 3D numpy array of class labels.
        overwrite (bool): Whether to overwrite the Zarr-2 file if it already exists.
        attrs (dict): A dictionary of attributes to save with the Zarr-2 file.

    Example usage:
        # Generate random class labels, with 0 as background
        labels = np.random.randint(0, 4, (128, 128, 128))
        save_numpy_labels_to_zarr('submission.zarr', 'test_volume', ['label1', 'label2', 'label3'], labels)
    """
    # Create a Zarr-2 file
    if not UPath(save_path).exists():
        os.makedirs(UPath(save_path).parent, exist_ok=True)
    logging.info(f"Saving to {save_path}")
    store = zarr.DirectoryStore(save_path)
    zarr_group = zarr.group(store)

    # Save the test volume group
    zarr_group.create_group(test_volume_name, overwrite=overwrite)

    # Save the labels
    for i, label_name in enumerate(label_name):
        logging.info(f"Saving {label_name}")
        ds = zarr_group[test_volume_name].create_dataset(
            label_name,
            data=(labels == i + 1),
            chunks=64,
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v

    logging.info("Done saving")


def save_numpy_class_arrays_to_zarr(
    save_path, test_volume_name, label_names, labels, mode="append", attrs=None
):
    """
    Save a list of 3D numpy arrays of binary or instance labels to a
    Zarr-2 file with the required structure.

    Args:
        save_path (str): The path to save the Zarr-2 file (ending with <filename>.zarr).
        test_volume_name (str): The name of the test volume.
        label_names (list): A list of label names corresponding to the list of 3D numpy arrays.
        labels (list): A list of 3D numpy arrays of binary labels.
        mode (str): The mode to use when saving the Zarr-2 file. Options are 'append' or 'overwrite'.
        attrs (dict): A dictionary of attributes to save with the Zarr-2 file.

    Example usage:
        label_names = ['label1', 'label2', 'label3']
        # Generate random binary volumes for each label
        labels = [np.random.randint(0, 2, (128, 128, 128)) for _ in range len(label_names)]
        save_numpy_binary_to_zarr('submission.zarr', 'test_volume', label_names, labels)

    """
    # Create a Zarr-2 file
    if not UPath(save_path).exists():
        os.makedirs(UPath(save_path).parent, exist_ok=True)
    logging.info(f"Saving to {save_path}")
    store = zarr.DirectoryStore(save_path)
    zarr_group = zarr.group(store)

    # Save the test volume group
    try:
        zarr_group.create_group(test_volume_name, overwrite=(mode == "overwrite"))
    except zarr.errors.ContainsGroupError:
        logging.info(f"Appending to existing group {test_volume_name}")

    # Save the labels
    for i, label_name in enumerate(label_names):
        logging.info(f"Saving {label_name}")
        ds = zarr_group[test_volume_name].create_dataset(
            label_name,
            data=labels[i],
            chunks=64,
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v

    logging.info("Done saving")


def zip_submission(zarr_path: str | UPath = SUBMISSION_PATH):
    """
    (Re-)Zip a submission zarr file.

    Args:
        zarr_path (str | UPath): The path to the submission zarr file (ending with `<filename>.zarr`). `.zarr` will be replaced with `.zip`.
    """
    zarr_path = UPath(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Submission zarr file not found at {zarr_path}")

    zip_path = zarr_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(zarr_path, followlinks=True):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure symlink targets are added as files
                if os.path.islink(file_path):
                    file_path = os.readlink(file_path)

                # Define the relative path in the zip archive
                arcname = os.path.relpath(file_path, zarr_path)
                zipf.write(file_path, arcname)

    logging.info(f"Zipped {zarr_path} to {zip_path}")

    return zip_path


def package_crop(crop, zarr_group, overwrite, input_search_path=PROCESSED_PATH):
    crop_path = (
        UPath(
            format_string(
                input_search_path, {"crop": f"crop{crop.id}", "dataset": crop.dataset}
            )
        )
        / crop.class_label
    )
    if not crop_path.exists():
        return f"Skipping {crop_path} as it does not exist."
    if f"crop{crop.id}" not in zarr_group:
        crop_group = zarr_group.create_group(f"crop{crop.id}")
    else:
        crop_group = zarr_group[f"crop{crop.id}"]

    try:
        logging.info(f"Scaling {crop_path} to {crop.voxel_size} nm")
        # Match the resolution, spatial position, and shape of the processed volume to the test volume
        image = match_crop_space(
            path=crop_path.path,
            class_label=crop.class_label,
            voxel_size=crop.voxel_size,
            shape=crop.shape,
            translation=crop.translation,
        )
        image = image.astype(np.uint8)
        # Save the processed labels to the submission zarr
        label_array = crop_group.create_dataset(
            crop.class_label,
            overwrite=overwrite,
            shape=crop.shape,
            dtype=image.dtype,
        )
        label_array[:] = image
        # Add the metadata
        label_array.attrs["voxel_size"] = crop.voxel_size
        label_array.attrs["translation"] = crop.translation
        label_array.attrs["shape"] = crop.shape

        return crop_path.path
    except Exception as e:
        error_msg = (
            f"Failed to package crop {crop.id} ({crop.class_label}) from {crop_path}. "
            f"Error: {type(e).__name__}: {e}"
        )
        logging.error(error_msg)
        return f"Error: {error_msg}"


def package_submission(
    input_search_path: str | UPath = PROCESSED_PATH,
    output_path: str | UPath = SUBMISSION_PATH,
    overwrite: bool = False,
    max_workers: int = os.cpu_count(),
):
    """
    Package a submission for the CellMap challenge. This will create a zarr file, combining all the processed volumes, and then zip it.

    Args:
        input_search_path (str): The base path to the processed volumes, with placeholders for dataset and crops.
        output_path (str | UPath): The path to save the submission zarr to. (ending with `<filename>.zarr`; `.zarr` will be appended if not present, and replaced with `.zip` when zipped).
        overwrite (bool): Whether to overwrite the submission zarr if it already exists.
        max_workers (int): The maximum number of workers to use for parallel processing. Defaults to the number of CPUs.
    """
    input_search_path = str(input_search_path)
    output_path = UPath(output_path)
    output_path = output_path.with_suffix(".zarr")

    # Create a zarr file to store the submission
    if not output_path.exists():
        os.makedirs(output_path.parent, exist_ok=True)
    store = zarr.DirectoryStore(output_path)
    zarr_group = zarr.group(store, overwrite=True)

    # Make groups for each test volume
    for crop in TEST_CROPS:
        if f"crop{crop.id}" not in zarr_group:
            crop_group = zarr_group.create_group(f"crop{crop.id}")

    # Find all the processed test volumes
    pool = ThreadPoolExecutor(max_workers)
    partial_package_crop = functools.partial(
        package_crop,
        zarr_group=zarr_group,
        overwrite=overwrite,
        input_search_path=input_search_path,
    )
    successful_crops = 0
    failed_crops = []
    skipped_crops = []
    for crop_path in tqdm(
        pool.map(partial_package_crop, TEST_CROPS),
        total=len(TEST_CROPS),
        dynamic_ncols=True,
        desc="Packaging crops...",
    ):
        if crop_path.lower().startswith("error:"):
            tqdm.write(crop_path)
            failed_crops.append(crop_path)
        elif "skipping" in crop_path.lower():
            tqdm.write(f"{crop_path} skipped.")
            skipped_crops.append(crop_path)
        else:
            tqdm.write(f"Packaged {crop_path}")
            successful_crops += 1

    logging.info(f"Packaged {successful_crops}/{len(TEST_CROPS)} crops.")
    if skipped_crops:
        logging.info(f"Skipped {len(skipped_crops)} crops (files did not exist).")
    if failed_crops:
        logging.error(f"Failed to package {len(failed_crops)} crops:")
        for error in failed_crops:
            logging.error(f"  {error}")

    logging.info(f"Saved submission to {output_path}")

    if successful_crops == 0:
        raise RuntimeError(
            f"No crops were packaged; submission zarr is empty. "
            f"Skipped {len(skipped_crops)}, failed {len(failed_crops)}."
        )

    logging.info("Zipping submission...")
    zip_submission(output_path)

    logging.info("Done packaging submission")
