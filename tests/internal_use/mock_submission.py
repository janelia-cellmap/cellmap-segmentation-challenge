import os

import numpy as np
import zarr
from tqdm import tqdm
from upath import UPath

import zarr.errors

import functools
from concurrent.futures import ThreadPoolExecutor

from cellmap_segmentation_challenge.config import (
    PROCESSED_PATH,
    SUBMISSION_PATH,
    TRUTH_PATH,
)
from cellmap_segmentation_challenge.utils import (
    TEST_CROPS,
    simulate_predictions_accuracy,
    simulate_predictions_iou,
    format_string,
)
from cellmap_segmentation_challenge.evaluate import (
    INSTANCE_CLASSES,
    zip_submission,
    match_crop_space,
)

CONFIGURED_ACCURACY = 0.6
CONFIGURED_IOU = 0.6


def mock_submission(
    input_search_path: str | UPath = (UPath(TRUTH_PATH) / "{crop}").path,
    output_path: str | UPath = SUBMISSION_PATH,
    overwrite: bool = True,
    max_workers: int = os.cpu_count(),
    configured_accuracy: float = CONFIGURED_ACCURACY,
    configured_iou: float = CONFIGURED_IOU,
):
    """
    Mock a submission by simulating errors in the processed volumes and packaging them into a submission zarr/zip.

    Args:
        input_search_path (str): The base path to the processed volumes, with placeholders for dataset and crops.
        output_path (str | UPath): The path to save the submission zarr to. (ending with `<filename>.zarr`; `.zarr` will be appended if not present, and replaced with `.zip` when zipped).
        overwrite (bool): Whether to overwrite the submission zarr if it already exists.
        max_workers (int): The maximum number of workers to use for parallel processing. Defaults to the number of CPUs.
        configured_accuracy (float): The configured accuracy to simulate errors with. Defaults to 0.6.
        configured_iou (float): The configured IOU to simulate errors with. Defaults to 0.6.
    """
    input_search_path = str(input_search_path)
    output_path = UPath(output_path)
    output_path = output_path.with_suffix(".zarr")

    # Create a zarr file to store the submission
    if not output_path.exists():
        os.makedirs(output_path.parent, exist_ok=True)
    store = zarr.DirectoryStore(output_path)
    zarr_group = zarr.group(store, overwrite=True)

    # Find all the processed test volumes
    pool = ThreadPoolExecutor(max_workers)
    partial_package_crop = functools.partial(
        mock_crop,
        zarr_group=zarr_group,
        overwrite=overwrite,
        input_search_path=input_search_path,
        configured_accuracy=configured_accuracy,
        configured_iou=configured_iou,
    )
    for crop_path in tqdm(
        pool.map(partial_package_crop, TEST_CROPS),
        total=len(TEST_CROPS),
        dynamic_ncols=True,
        desc="Mocking crops...",
    ):
        tqdm.write(f"Packaged {crop_path}")

    print(f"Saved mock submission to {output_path}")

    print("Zipping mock submission...")
    zip_submission(output_path)

    print(
        f"Done packaging mock submission with configured accuracy: {CONFIGURED_ACCURACY} and configured IOU: {CONFIGURED_IOU}"
    )


def mock_crop(
    crop,
    zarr_group,
    overwrite,
    input_search_path=PROCESSED_PATH,
    configured_accuracy=CONFIGURED_ACCURACY,
    configured_iou=CONFIGURED_IOU,
):
    input_path = format_string(
        input_search_path,
        format_kwargs={"crop": f"crop{crop.id}", "dataset": crop.dataset},
    )
    crop_path = UPath(input_path) / crop.class_label
    if not crop_path.exists():
        return f"Skipping {crop_path} as it does not exist."
    if f"crop{crop.id}" not in zarr_group:
        crop_group = zarr_group.create_group(f"crop{crop.id}")
    else:
        crop_group = zarr_group[f"crop{crop.id}"]

    print(f"Scaling {crop_path} to {crop.voxel_size} nm")
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

    # Add errors to the ground truth to simulate a submission
    if crop.class_label in INSTANCE_CLASSES:
        # Add errors to the instance segmentation
        image = simulate_predictions_accuracy(image, configured_accuracy)
    else:
        # Add errors to the semantic segmentation
        image = simulate_predictions_iou(image, configured_iou)

    label_array[:] = image
    # Add the metadata
    label_array.attrs["voxel_size"] = crop.voxel_size
    label_array.attrs["translation"] = crop.translation
    label_array.attrs["shape"] = crop.shape

    return crop_path


if __name__ == "__main__":
    mock_submission()
