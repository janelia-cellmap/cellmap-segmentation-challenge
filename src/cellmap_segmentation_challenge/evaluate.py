import argparse
import json
import os
import zipfile

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import dice  # , jaccard
from skimage.measure import label as relabel
from skimage.transform import rescale
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
from upath import UPath

from cellmap_data import CellMapImage
import zarr.errors

from .config import PROCESSED_PATH, SUBMISSION_PATH, TRUTH_PATH
from .utils import TEST_CROPS, TEST_CROPS_DICT


INSTANCE_CLASSES = [
    "nuc",
    "vim",
    "ves",
    "endo",
    "lyso",
    "ld",
    "perox",
    "mito",
    "np",
    "mt",
    "cell",
    "instance",
]


HAUSDORFF_DISTANCE_MAX = np.inf


def unzip_file(zip_path):
    """
    Unzip a zip file to a specified directory.

    Args:
        zip_path (str): The path to the zip file.

    Example usage:
        unzip_file('submission.zip')
    """
    saved_path = UPath(zip_path).with_suffix(".zarr").path
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(saved_path)
        print(f"Unzipped {zip_path} to {saved_path}")

    return UPath(saved_path)


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
    print(f"Saving to {save_path}")
    store = zarr.DirectoryStore(save_path)
    zarr_group = zarr.group(store)

    # Save the test volume group
    zarr_group.create_group(test_volume_name, overwrite=overwrite)

    # Save the labels
    for i, label_name in enumerate(label_name):
        print(f"Saving {label_name}")
        ds = zarr_group[test_volume_name].create_dataset(
            label_name,
            data=(labels == i + 1),
            chunks=64,
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v

    print("Done saving")


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
    print(f"Saving to {save_path}")
    store = zarr.DirectoryStore(save_path)
    zarr_group = zarr.group(store)

    # Save the test volume group
    try:
        zarr_group.create_group(test_volume_name, overwrite=(mode == "overwrite"))
    except zarr.errors.ContainsGroupError:
        print(f"Appending to existing group {test_volume_name}")

    # Save the labels
    for i, label_name in enumerate(label_names):
        print(f"Saving {label_name}")
        ds = zarr_group[test_volume_name].create_dataset(
            label_name,
            data=labels[i],
            chunks=64,
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v

    print("Done saving")


def resize_array(arr, target_shape, pad_value=0):
    """
    Resize an array to a target shape by padding or cropping as needed.

    Parameters:
        arr (np.ndarray): Input array to resize.
        target_shape (tuple): Desired shape for the output array.
        pad_value (int, float, etc.): Value to use for padding if the array is smaller than the target shape.

    Returns:
        np.ndarray: Resized array with the specified target shape.
    """
    arr_shape = arr.shape
    resized_arr = arr

    # Pad if the array is smaller than the target shape
    pad_width = []
    for i in range(len(target_shape)):
        if arr_shape[i] < target_shape[i]:
            # Padding needed: calculate amount for both sides
            pad_before = (target_shape[i] - arr_shape[i]) // 2
            pad_after = target_shape[i] - arr_shape[i] - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            # No padding needed for this dimension
            pad_width.append((0, 0))

    if any(pad > 0 for pads in pad_width for pad in pads):
        resized_arr = np.pad(
            resized_arr, pad_width, mode="constant", constant_values=pad_value
        )

    # Crop if the array is larger than the target shape
    slices = []
    for i in range(len(target_shape)):
        if arr_shape[i] > target_shape[i]:
            # Calculate cropping slices to center the crop
            start = (arr_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            slices.append(slice(start, end))
        else:
            # No cropping needed for this dimension
            slices.append(slice(None))

    return resized_arr[tuple(slices)]


def hausdorff_distance(
    image0, image1, voxel_size, max_distance=np.inf, method="standard"
):
    """Calculate the Hausdorff distance between nonzero elements of given images.

    Modified from the `scipy.spatial.distance.hausdorff` function by Jeff Rhoades (2024) to handle non-isotropic resolutions.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    voxel_size : tuple
        The size of a voxel in each dimension. Assumes the same resolution for both images.
    max_distance : float, optional
        The maximum distance to consider. Default is infinity.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.

    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidean distance.

    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155

    """

    if method not in ("standard", "modified"):
        raise ValueError(f"unrecognized method {method}")

    a_points = np.argwhere(image0)
    b_points = np.argwhere(image1)

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    # Scale the points by the voxel size
    a_points = a_points * np.array(voxel_size)
    b_points = b_points * np.array(voxel_size)

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1, distance_upper_bound=max_distance)[0],
        cKDTree(b_points).query(a_points, k=1, distance_upper_bound=max_distance)[0],
    )

    if method == "standard":  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == "modified":  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))


def score_instance(
    pred_label, truth_label, voxel_size, hausdorff_distance_max=HAUSDORFF_DISTANCE_MAX
) -> dict[str, float]:
    """
    Score a single instance label volume against the ground truth instance label volume.

    Args:
        pred_label (np.ndarray): The predicted instance label volume.
        truth_label (np.ndarray): The ground truth instance label volume.
        voxel_size (tuple): The size of a voxel in each dimension.
        hausdorff_distance_max (float): The maximum distance to consider for the Hausdorff distance.

    Returns:
        dict: A dictionary of scores for the instance label volume.

    Example usage:
        scores = score_instance(pred_label, truth_label)
    """
    # Relabel the predicted instance labels to be consistent with the ground truth instance labels
    print("Scoring instance segmentation...")
    pred_label = relabel(pred_label, connectivity=len(pred_label.shape))
    # pred_label = label(pred_label)

    # Construct the cost matrix for Hungarian matching
    pred_ids = np.unique(pred_label)
    truth_ids = np.unique(truth_label)
    cost_matrix = np.zeros((len(truth_ids), len(pred_ids)))
    bar = tqdm(pred_ids, desc="Computing cost matrix", leave=True, dynamic_ncols=True)
    for j, pred_id in enumerate(bar):
        if pred_id == 0:
            # Don't score the background
            continue
        pred_mask = pred_label == pred_id
        these_truth_ids = np.unique(truth_label[pred_mask])
        truth_indices = [
            np.argmax(truth_ids == truth_id) for truth_id in these_truth_ids
        ]
        for i, truth_id in zip(truth_indices, these_truth_ids):
            if truth_id == 0:
                # Don't score the background
                continue
            truth_mask = truth_label == truth_id
            cost_matrix[i, j] = jaccard_score(truth_mask.flatten(), pred_mask.flatten())

    # Match the predicted instances to the ground truth instances
    row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)

    # Contruct the volume for the matched instances
    matched_pred_label = np.zeros_like(pred_label)
    for i, j in tqdm(
        zip(col_inds, row_inds), desc="Relabeled matched instances", dynamic_ncols=True
    ):
        if pred_ids[i] == 0 or truth_ids[j] == 0:
            # Don't score the background
            continue
        pred_mask = pred_label == pred_ids[i]
        matched_pred_label[pred_mask] = truth_ids[j]

    hausdorff_distances = []
    for truth_id in tqdm(
        truth_ids, desc="Computing Hausdorff distances", dynamic_ncols=True
    ):
        if truth_id == 0:
            # Don't score the background
            continue
        h_dist = hausdorff_distance(
            truth_label == truth_id,
            matched_pred_label == truth_id,
            voxel_size,
        )
        h_dist = min(h_dist, hausdorff_distance_max)
        hausdorff_distances.append(h_dist)

    # Compute the scores
    accuracy = accuracy_score(truth_label.flatten(), matched_pred_label.flatten())
    hausdorff_dist = np.mean(hausdorff_distances) if hausdorff_distances else 0
    normalized_hausdorff_dist = 1.01 ** (
        -hausdorff_dist / np.linalg.norm(voxel_size)
    )  # normalize Hausdorff distance to [0, 1] using the maximum distance represented by a voxel. 32 is abritrary chosen to have a reasonable range
    combined_score = (accuracy * normalized_hausdorff_dist) ** 0.5
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    print(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    print(f"Combined Score: {combined_score:.4f}")
    return {
        "accuracy": accuracy,
        "hausdorff_distance": hausdorff_dist,
        "normalized_hausdorff_distance": normalized_hausdorff_dist,
        "combined_score": combined_score,
    }


def score_semantic(pred_label, truth_label) -> dict[str, float]:
    """
    Score a single semantic label volume against the ground truth semantic label volume.

    Args:
        pred_label (np.ndarray): The predicted semantic label volume.
        truth_label (np.ndarray): The ground truth semantic label volume.

    Returns:
        dict: A dictionary of scores for the semantic label volume.

    Example usage:
        scores = score_semantic(pred_label, truth_label)
    """
    print("Scoring semantic segmentation...")
    # Flatten the label volumes and convert to binary
    pred_label = (pred_label > 0.0).flatten()
    truth_label = (truth_label > 0.0).flatten()
    # Compute the scores
    dice_score = 1 - dice(truth_label, pred_label)
    scores = {
        "iou": jaccard_score(truth_label, pred_label, zero_division=1),
        "dice_score": dice_score if not np.isnan(dice_score) else 1,
    }

    print(f"IoU: {scores['iou']:.4f}")
    print(f"Dice Score: {scores['dice_score']:.4f}")

    return scores


def score_label(
    pred_label_path, truth_path=TRUTH_PATH, instance_classes=INSTANCE_CLASSES
) -> dict[str, float]:
    """
    Score a single label volume against the ground truth label volume.

    Args:
        pred_label_path (str): The path to the predicted label volume.

    Returns:
        dict: A dictionary of scores for the label volume.

    Example usage:
        scores = score_label('pred.zarr/test_volume/label1')
    """
    print(f"Scoring {pred_label_path}...")
    truth_path = UPath(truth_path)
    # Load the predicted and ground truth label volumes
    label_name = UPath(pred_label_path).name
    crop_name = UPath(pred_label_path).parent.name
    truth_label_path = (truth_path / crop_name / label_name).path
    truth_label_ds = zarr.open(truth_label_path, mode="r")
    truth_label = truth_label_ds[:]
    crop = TEST_CROPS_DICT[int(crop_name.removeprefix("crop")), label_name]
    pred_label = match_crop_space(
        pred_label_path,
        label_name,
        crop.voxel_size,
        crop.shape,
        crop.translation,
    )

    mask_path = truth_path / crop_name / f"{label_name}_mask"
    if mask_path.exists():
        # Mask out uncertain regions resulting from low-res ground truth annotations
        print(f"Masking {label_name} with {mask_path}...")
        mask = zarr.open(mask_path.path, mode="r")[:]
        pred_label = pred_label * mask
        truth_label = truth_label * mask

    # Compute the scores
    if label_name in instance_classes:
        results = score_instance(pred_label, truth_label, crop.voxel_size)
    else:
        results = score_semantic(pred_label, truth_label)
    results["num_voxels"] = int(np.prod(truth_label.shape))
    results["voxel_size"] = crop.voxel_size
    results["is_missing"] = False
    return results


def score_volume(
    pred_volume_path, truth_path=TRUTH_PATH, instance_classes=INSTANCE_CLASSES
) -> dict[str, dict[str, float]]:
    """
    Score a single volume against the ground truth volume.

    Args:
        pred_volume_path (str): The path to the predicted volume.

    Returns:
        dict: A dictionary of scores for the volume.

    Example usage:
        scores = score_volume('pred.zarr/test_volume')
    """
    print(f"Scoring {pred_volume_path}...")
    pred_volume_path = UPath(pred_volume_path)
    truth_path = UPath(truth_path)

    # Find labels to score
    pred_labels = [a for a in zarr.open(pred_volume_path.path, mode="r").array_keys()]

    volume_name = pred_volume_path.name
    truth_labels = [
        a for a in zarr.open((truth_path / volume_name).path, mode="r").array_keys()
    ]

    found_labels = list(set(pred_labels) & set(truth_labels))
    missing_labels = list(set(truth_labels) - set(pred_labels))

    # Score each label
    scores = {
        label: score_label(
            pred_volume_path / label,
            truth_path=truth_path,
            instance_classes=instance_classes,
        )
        for label in found_labels
    }
    scores.update(
        {
            label: (
                {
                    "accuracy": 0,
                    "hausdorff_distance": 0,
                    "normalized_hausdorff_distance": 0,
                    "combined_score": 0,
                    "num_voxels": int(
                        np.prod(
                            zarr.open(
                                (truth_path / volume_name / label).path, mode="r"
                            ).shape
                        )
                    ),
                    "voxel_size": zarr.open(
                        (truth_path / volume_name / label).path, mode="r"
                    ).attrs["voxel_size"],
                    "is_missing": True,
                }
                if label in instance_classes
                else {
                    "iou": 0,
                    "dice_score": 0,
                    "num_voxels": int(
                        np.prod(
                            zarr.open(
                                (truth_path / volume_name / label).path, mode="r"
                            ).shape
                        )
                    ),
                    "voxel_size": zarr.open(
                        (truth_path / volume_name / label).path, mode="r"
                    ).attrs["voxel_size"],
                    "is_missing": True,
                }
            )
            for label in missing_labels
        }
    )
    print(f"Missing labels: {missing_labels}")

    return scores


def missing_volume_score(
    truth_volume_path, instance_classes=INSTANCE_CLASSES
) -> dict[str, dict[str, float]]:
    """
    Score a missing volume as 0's, congruent with the score_volume function.

    Args:
        truth_volume_path (str): The path to the ground truth volume.

    Returns:
        dict: A dictionary of scores for the volume.

    Example usage:
        scores = missing_volume_score('truth.zarr/test_volume')
    """
    print(f"Scoring missing volume {truth_volume_path}...")
    truth_volume_path = UPath(truth_volume_path)

    # Find labels to score
    truth_labels = [a for a in zarr.open(truth_volume_path.path, mode="r").array_keys()]

    # Score each label
    scores = {
        label: (
            {
                "accuracy": 0.0,
                "hausdorff_distance": 0.0,
                "normalized_hausdorff_distance": 0.0,
                "combined_score": 0.0,
                "num_voxels": int(
                    np.prod(zarr.open((truth_volume_path / label).path, mode="r").shape)
                ),
                "voxel_size": zarr.open(
                    (truth_volume_path / label).path, mode="r"
                ).attrs["voxel_size"],
                "is_missing": True,
            }
            if label in instance_classes
            else {
                "iou": 0.0,
                "dice_score": 0.0,
                "num_voxels": int(
                    np.prod(zarr.open((truth_volume_path / label).path, mode="r").shape)
                ),
                "voxel_size": zarr.open(
                    (truth_volume_path / label).path, mode="r"
                ).attrs["voxel_size"],
                "is_missing": True,
            }
        )
        for label in truth_labels
    }

    return scores


def combine_scores(scores, include_missing=True, instance_classes=INSTANCE_CLASSES):
    """
    Combine scores across volumes, normalizing by the number of voxels.

    Args:
        scores (dict): A dictionary of scores for each volume, as returned by `score_volume`.
        include_missing (bool): Whether to include missing volumes in the combined scores.
        instance_classes (list): A list of instance classes.

    Returns:
        dict: A dictionary of combined scores across all volumes.

    Example usage:
        combined_scores = combine_scores(scores)
    """

    # Combine label scores across volumes, normalizing by the number of voxels
    print(f"Combining label scores...")
    scores = scores.copy()
    label_scores = {}
    total_volumes = {}
    for these_scores in scores.values():
        for label, this_score in these_scores.items():
            print(this_score)
            if this_score["is_missing"] and not include_missing:
                continue
            total_volume = np.prod(this_score["voxel_size"]) * this_score["num_voxels"]
            if label in instance_classes:
                if label not in label_scores:
                    label_scores[label] = {
                        "accuracy": 0,
                        "hausdorff_distance": 0,
                        "normalized_hausdorff_distance": 0,
                        "combined_score": 0,
                    }
                    total_volumes[label] = 0
                label_scores[label]["accuracy"] += this_score["accuracy"] * total_volume
                label_scores[label]["hausdorff_distance"] += (
                    this_score["hausdorff_distance"] * total_volume
                )
                label_scores[label]["normalized_hausdorff_distance"] += (
                    this_score["normalized_hausdorff_distance"] * total_volume
                )
                label_scores[label]["combined_score"] += (
                    this_score["combined_score"] * total_volume
                )
                total_volumes[label] += total_volume
            else:
                if label not in label_scores:
                    label_scores[label] = {"iou": 0, "dice_score": 0}
                    total_volumes[label] = 0
                label_scores[label]["iou"] += this_score["iou"] * total_volume
                label_scores[label]["dice_score"] += (
                    this_score["dice_score"] * total_volume
                )
                total_volumes[label] += total_volume

    # Normalize back to the total number of voxels
    for label in label_scores:
        if label in instance_classes:
            label_scores[label]["accuracy"] /= total_volumes[label]
            label_scores[label]["hausdorff_distance"] /= total_volumes[label]
            label_scores[label]["normalized_hausdorff_distance"] /= total_volumes[label]
            label_scores[label]["combined_score"] /= total_volumes[label]
        else:
            label_scores[label]["iou"] /= total_volumes[label]
            label_scores[label]["dice_score"] /= total_volumes[label]
    scores["label_scores"] = label_scores

    # Compute the overall score
    print("Computing overall scores...")
    overall_instance_scores = []
    overall_semantic_scores = []
    for label in label_scores:
        if label in instance_classes:
            overall_instance_scores += [label_scores[label]["combined_score"]]
        else:
            overall_semantic_scores += [label_scores[label]["iou"]]
    scores["overall_instance_score"] = np.mean(overall_instance_scores)
    scores["overall_semantic_score"] = np.mean(overall_semantic_scores)
    scores["overall_score"] = (
        scores["overall_instance_score"] * scores["overall_semantic_score"]
    ) ** 0.5  # geometric mean

    return scores


def score_submission(
    submission_path=UPath(SUBMISSION_PATH).with_suffix(".zip").path,
    result_file=None,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
):
    """
    Score a submission against the ground truth data.

    Args:
        submission_path (str): The path to the zipped submission Zarr-2 file.
        result_file (str): The path to save the scores.

    Returns:
        dict: A dictionary of scores for the submission.

    Example usage:
        scores = score_submission('submission.zip')

    The results json is a dictionary with the following structure:
    {
        "volume" (the name of the ground truth volume): {
            "label" (the name of the predicted class): {
                (For semantic segmentation)
                    "iou": (the intersection over union score),
                    "dice_score": (the dice score),

                OR

                (For instance segmentation)
                    "accuracy": (the accuracy score),
                    "haussdorf_distance": (the haussdorf distance),
                    "normalized_haussdorf_distance": (the normalized haussdorf distance),
                    "combined_score": (the geometric mean of the accuracy and normalized haussdorf distance),
            }
            "num_voxels": (the number of voxels in the ground truth volume),
        }
        "label_scores": {
            (the name of the predicted class): {
                (For semantic segmentation)
                    "iou": (the mean intersection over union score),
                    "dice_score": (the mean dice score),

                OR

                (For instance segmentation)
                    "accuracy": (the mean accuracy score),
                    "haussdorf_distance": (the mean haussdorf distance),
                    "combined_score": (the mean geometric mean of the accuracy and haussdorf distance),
            }
        "overall_score": (the mean of the combined scores across all classes),
    }
    """
    print(f"Scoring {submission_path}...")
    # Unzip the submission
    submission_path = unzip_file(submission_path)

    # Find volumes to score
    print(f"Scoring volumes in {submission_path}...")
    pred_volumes = [d.name for d in UPath(submission_path).glob("*") if d.is_dir()]
    truth_path = UPath(truth_path)
    print(f"Volumes: {pred_volumes}")
    print(f"Truth path: {truth_path}")
    truth_volumes = [d.name for d in truth_path.glob("*") if d.is_dir()]
    print(f"Truth volumes: {truth_volumes}")

    found_volumes = list(
        set(pred_volumes) & set(truth_volumes)
    )  # TODO: Score all volumes, and just return 0 scores for missing volumes
    missing_volumes = list(set(truth_volumes) - set(pred_volumes))
    if len(found_volumes) == 0:
        raise ValueError(
            "No volumes found to score. Make sure the submission is formatted correctly."
        )
    print(f"Scoring volumes: {found_volumes}")
    if len(missing_volumes) > 0:
        print(f"Missing volumes: {missing_volumes}")
        print("Scoring missing volumes as 0's")

    # Score each volume
    scores = {
        volume: score_volume(
            submission_path / volume,
            truth_path=truth_path,
            instance_classes=instance_classes,
        )
        for volume in found_volumes
    }
    scores.update(
        {
            volume: missing_volume_score(
                truth_path / volume, instance_classes=instance_classes
            )
            for volume in missing_volumes
        }
    )

    # Combine label scores across volumes, normalizing by the number of voxels
    all_scores = combine_scores(
        scores, include_missing=True, instance_classes=instance_classes
    )
    found_scores = combine_scores(
        scores, include_missing=False, instance_classes=instance_classes
    )

    print("Scores combined across all test volumes:")
    print(f"\tOverall Instance Score: {all_scores['overall_instance_score']:.4f}")
    print(f"\tOverall Semantic Score: {all_scores['overall_semantic_score']:.4f}")
    print(f"\tOverall Score: {all_scores['overall_score']:.4f}")

    print("Scores combined across test volumes with data submitted:")
    print(f"\tOverall Instance Score: {found_scores['overall_instance_score']:.4f}")
    print(f"\tOverall Semantic Score: {found_scores['overall_semantic_score']:.4f}")
    print(f"\tOverall Score: {found_scores['overall_score']:.4f}")

    # Save the scores
    if result_file:
        print(f"Saving collected scores to {result_file}...")
        with open(result_file, "w") as f:
            json.dump(all_scores, f)

        found_result_file = str(result_file).replace(
            UPath(result_file).suffix, "_submitted_only" + UPath(result_file).suffix
        )
        print(f"Saving scores for only submitted data to {found_result_file}...")
        with open(found_result_file, "w") as f:
            json.dump(found_scores, f)

    else:
        return all_scores


def package_submission(
    input_search_path: str | UPath = PROCESSED_PATH,
    output_path: str | UPath = SUBMISSION_PATH,
    overwrite: bool = False,
):
    """
    Package a submission for the CellMap challenge. This will create a zarr file, combining all the processed volumes, and then zip it.

    Args:
        input_search_path (str): The base path to the processed volumes, with placeholders for dataset and crops.
        output_path (str | UPath): The path to save the submission zarr to. (ending with `<filename>.zarr`; `.zarr` will be appended if not present, and replaced with `.zip` when zipped).
        overwrite (bool): Whether to overwrite the submission zarr if it already exists.
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
    for crop in TEST_CROPS:
        crop_path = (
            UPath(input_search_path.format(dataset=crop.dataset, crop=f"crop{crop.id}"))
            / crop.class_label
        )
        if not crop_path.exists():
            print(f"Skipping {crop_path} as it does not exist.")
            continue
        if f"crop{crop.id}" not in zarr_group:
            crop_group = zarr_group.create_group(f"crop{crop.id}")
        else:
            crop_group = zarr_group[f"crop{crop.id}"]

        # Rescale the processed volumes to match the expected submission resolution if required
        label_array = crop_group.create_dataset(
            crop.class_label,
            overwrite=overwrite,
            shape=crop.shape,
        )
        print(f"Scaling {crop_path} to {crop.voxel_size} nm")
        # Match the resolution of the processed volume to the test volume
        image = match_crop_space(
            path=crop_path.path,
            class_label=crop.class_label,
            voxel_size=crop.voxel_size,
            shape=crop.shape,
            translation=crop.translation,
        )
        # Save the processed labels to the submission zarr
        label_array[:] = image
        # Add the metadata
        label_array.attrs["voxel_size"] = crop.voxel_size
        label_array.attrs["translation"] = crop.translation

    print(f"Saved submission to {output_path}")

    print("Zipping submission...")
    zip_submission(output_path)

    print("Done packaging submission")


def match_crop_space(path, class_label, voxel_size, shape, translation) -> np.ndarray:
    """
    Match the resolution of a zarr array to a target resolution and shape, resampling as necessary with interpolation dependent on the class label. Instance segmentations will be resampled with nearest neighbor interpolation, while semantic segmentations will be resampled with linear interpolation and then thresholded.

    Args:
        path (str | UPath): The path to the zarr array to be adjusted. The zarr can be an OME-NGFF multiscale zarr file, or a traditional single scale formatted zarr.
        class_label (str): The class label of the array.
        voxel_size (tuple): The target voxel size.
        shape (tuple): The target shape.
        translation (tuple): The translation (i.e. offset) of the array in world units.

    Returns:
        np.ndarray: The rescaled array.
    """
    ds = zarr.open(str(path), mode="r")
    if "multiscales" in ds.attrs:
        # Handle multiscale zarr files
        _image = CellMapImage(
            path=path,
            target_class=class_label,
            target_scale=voxel_size,
            target_voxel_shape=shape,
            pad=True,
            pad_value=0,
        )
        path = UPath(path) / _image.scale_level
        for attr in ds.attrs["multiscales"][0]["datasets"]:
            if attr["path"] == _image.scale_level:
                for transform in attr["coordinateTransformations"]:
                    if transform["type"] == "translation":
                        input_translation = transform["translation"]
                    elif transform["type"] == "scale":
                        input_voxel_size = transform["scale"]
                break
        ds = zarr.open(path.path, mode="r")
    elif (
        ("voxel_size" in ds.attrs)
        or ("resolution" in ds.attrs)
        or ("scale" in ds.attrs)
    ) or (("translation" in ds.attrs) or ("offset" in ds.attrs)):
        # Handle single scale zarr files
        if "voxel_size" in ds.attrs:
            input_voxel_size = ds.attrs["voxel_size"]
        elif "resolution" in ds.attrs:
            input_voxel_size = ds.attrs["resolution"]
        elif "scale" in ds.attrs:
            input_voxel_size = ds.attrs["scale"]
        else:
            input_voxel_size = None

        if "translation" in ds.attrs:
            input_translation = ds.attrs["translation"]
        elif "offset" in ds.attrs:
            input_translation = ds.attrs["offset"]
        else:
            input_translation = None
    else:
        print(f"Could not find voxel size and translation for {path}")
        print(
            "Assuming voxel size matches target voxel size and will crop to target shape centering the volume."
        )
        image = ds[:]
        # Crop the array if necessary
        if any(s1 != s2 for s1, s2 in zip(image.shape, shape)):
            return resize_array(image, shape)  # type: ignore
        return image  # type: ignore

    # Load the array
    image = ds[:]

    # Rescale the array if necessary
    if input_voxel_size is not None and any(
        r1 != r2 for r1, r2 in zip(input_voxel_size, voxel_size)
    ):
        if class_label in INSTANCE_CLASSES:
            image = rescale(
                image, np.divide(input_voxel_size, voxel_size), order=0, mode="constant"
            )
        else:
            image = rescale(
                image,
                np.divide(input_voxel_size, voxel_size),
                order=1,
                mode="constant",
                preserve_range=True,
            )
            image = image > 0.5

    if input_translation is not None:
        # Calculate the relative offset
        adjusted_input_translation = (
            np.array(input_translation) // np.array(voxel_size)
        ) * np.array(voxel_size)

        # Positive relative offset is the amount to crop from the start, negative is the amount to pad at the start
        relative_offset = (
            abs(np.subtract(adjusted_input_translation, translation))
            // np.array(voxel_size)
            * np.sign(np.subtract(adjusted_input_translation, translation))
        )
    else:
        # TODO: Handle the case where the translation is not provided
        relative_offset = np.zeros(len(shape))

    # Translate and crop the array if necessary
    if any(offset != 0 for offset in relative_offset) or any(
        s1 != s2 for s1, s2 in zip(image.shape, shape)
    ):
        print(
            f"Translating and cropping {path} to {shape} with offset {relative_offset}"
        )
        # Make destination array
        result = np.zeros(shape, dtype=image.dtype)

        # Calculate the slices for the source and destination arrays
        input_slices = []
        output_slices = []
        for i in range(len(shape)):
            if relative_offset[i] < 0:
                # Crop from the start
                input_start = abs(relative_offset[i])
                output_start = 0
                input_end = min(input_start + shape[i], image.shape[i])
                input_length = input_end - input_start
                output_end = output_start + input_length
            else:
                # Pad at the start
                input_start = 0
                output_start = relative_offset[i]
                output_end = min(shape[i], image.shape[i])
                input_length = output_end - output_start
                input_end = input_length

            if input_length <= 0:
                print("WARNING: Cropping to proper offset resulted in empty volume.")
                print(f"\tInput shape: {image.shape}, Output shape: {shape}")
                print(f"\tInput offset: {input_start}, Output offset: {output_start}")
                return result

            input_slices.append(slice(int(input_start), int(input_end)))
            output_slices.append(slice(int(output_start), int(output_end)))

        # Copy the data
        result[*output_slices] = image[*input_slices]
        return result
    else:
        return image


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

    print(f"Zipped {zarr_path} to {zip_path}")

    return zip_path


if __name__ == "__main__":
    # When called on the commandline, evaluate the submission
    # example usage: python evaluate.py submission.zip
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "submission_file", help="Path to submission zip file to score"
    )
    argparser.add_argument(
        "result_file",
        nargs="?",
        help="If provided, store submission results in this file. Else print them to stdout",
    )
    argparser.add_argument(
        "--truth-path", default=TRUTH_PATH, help="Path to zarr containing ground truth"
    )
    args = argparser.parse_args()

    score_submission(args.submission_file, args.result_file, args.truth_path)
