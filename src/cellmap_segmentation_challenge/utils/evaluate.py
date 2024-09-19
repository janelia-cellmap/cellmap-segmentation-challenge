"""
Submission requirements:
1. The submission should be a single zip file containing a single Zarr-2 file with the following structure:
   - submission.zarr
     - /<test_volume_name>
        - /<label_name>
2. The names of the test volumes and labels should match the names of the test volumes and labels in the test data.
3. Each label volume should be either A) a 3D binary volume with the same shape and scale as the corresponding test volume, or B) instance IDs per object.
4. The scale for all volumes is 8x8x8 nm/voxel, except as otherwise specified.

Assuming your data is already 8x8x8nm/voxel, you can convert the submission to the required format using the following convenience functions:

- For converting a single 3D numpy array of class labels to a Zarr-2 file, use the following function:
  `cellmap_segmentation_challenge.utils.evaluate.save_numpy_labels_to_zarr`
Note: The class labels should start from 1, with 0 as background.

- For converting a list of 3D numpy arrays of binary or instance labels to a Zarr-2 file, use the following function:
  `cellmap_segmentation_challenge.utils.evaluate.save_numpy_binary_to_zarr`
Note: The instance labels, if used, should be unique IDs per object, with 0 as background.

The arguments for both functions are the same:
- `submission_path`: The path to save the Zarr-2 file (ending with <filename>.zarr).
- `test_volume_name`: The name of the test volume.
- `label_names`: A list of label names corresponding to the list of 3D numpy arrays or the number of the class labels (0 is always assumed to be background).
- `labels`: A list of 3D numpy arrays of binary labels or a single 3D numpy array of class labels.
- `overwrite`: A boolean flag to overwrite the Zarr-2 file if it already exists.

To zip the Zarr-2 file, you can use the following command:
`zip -r submission.zip submission.zarr`

To submit the zip file, upload it to the challenge platform.
"""

import json
import sys
import zipfile
import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    jaccard_score,
    accuracy_score,
)
from skimage.metrics import hausdorff_distance
from scipy.spatial.distance import dice as dice_score

import zarr
import os
from upath import UPath


INSTANCE_CLASSES = ["mito", "nuc"]
INSTANCE_THRESHOLD = 0.5
HAUSDORFF_DISTANCE_MAX = np.inf

# TODO: REPLACE WITH THE GROUND TRUTH LABEL VOLUME PATH
TRUTH_PATH = "data/ground_truth.zarr"


def unzip_file(zip_path):
    """
    Unzip a zip file to a specified directory.

    Args:
        zip_path (str): The path to the zip file.

    Example usage:
        unzip_file('submission.zip')
    """
    name = UPath(zip_path).name
    extract_path = UPath(zip_path).parent / name.split(".")[0]
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path


def save_numpy_labels_to_zarr(
    submission_path, test_volume_name, label_name, labels, overwrite=False
):
    """
    Save a single 3D numpy array of class labels to a
    Zarr-2 file with the required structure.

    Args:
        submission_path (str): The path to save the Zarr-2 file (ending with <filename>.zarr).
        test_volume_name (str): The name of the test volume.
        label_names (str): The names of the labels.
        labels (np.ndarray): A 3D numpy array of class labels.

    Example usage:
        # Generate random class labels, with 0 as background
        labels = np.random.randint(0, 4, (128, 128, 128))
        save_numpy_labels_to_zarr('submission.zarr', 'test_volume', ['label1', 'label2', 'label3'], labels)
    """
    # Create a Zarr-2 file
    if not UPath(submission_path).exists():
        os.makedirs(UPath(submission_path).parent)
        store = zarr.DirectoryStore(submission_path)
        zarr_group = zarr.group(store)

    # Save the test volume group
    zarr_group.create_group(test_volume_name, overwrite=overwrite)

    # Save the labels
    for i, label_name in enumerate(label_name):
        zarr_group[test_volume_name].create_dataset(
            label_name,
            data=(labels == i + 1),
            chunks=(64, 64, 64),
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )


def save_numpy_binary_to_zarr(
    submission_path, test_volume_name, label_names, labels, overwrite=False
):
    """
    Save a list of 3D numpy arrays of binary labels to a
    Zarr-2 file with the required structure.

    Args:
        submission_path (str): The path to save the Zarr-2 file (ending with <filename>.zarr).
        test_volume_name (str): The name of the test volume.
        label_names (list): A list of label names corresponding to the list of 3D numpy arrays.
        labels (list): A list of 3D numpy arrays of binary labels.

    Example usage:
        label_names = ['label1', 'label2', 'label3']
        # Generate random binary volumes for each label
        labels = [np.random.randint(0, 2, (128, 128, 128)) for _ in range len(label_names)]
        save_numpy_binary_to_zarr('submission.zarr', 'test_volume', label_names, labels)

    """
    # Create a Zarr-2 file
    if not UPath(submission_path).exists():
        os.makedirs(UPath(submission_path).parent)
        store = zarr.DirectoryStore(submission_path)
        zarr_group = zarr.group(store)

    # Save the test volume group
    zarr_group.create_group(test_volume_name, overwrite=overwrite)

    # Save the labels
    for i, label_name in enumerate(label_names):
        zarr_group[test_volume_name].create_dataset(
            label_name,
            data=labels[i],
            chunks=(64, 64, 64),
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )


def score_instance(pred_label, truth_label) -> dict[str, float]:
    """
    Score a single instance label volume against the ground truth instance label volume.

    Args:
        pred_label (np.ndarray): The predicted instance label volume.
        truth_label (np.ndarray): The ground truth instance label volume.

    Returns:
        dict: A dictionary of scores for the instance label volume.

    Example usage:
        scores = score_instance(pred_label, truth_label)
    """
    # Relabel the predicted instance labels to be consistent with the ground truth instance labels
    pred_label, _ = label(pred_label, structure=np.ones((3, 3, 3)))

    # Construct the cost matrix for Hungarian matching
    pred_ids = np.unique(pred_label)
    truth_ids = np.unique(truth_label)
    cost_matrix = np.zeros((len(truth_ids), len(pred_ids)))
    for i, pred_id in enumerate(pred_ids):
        if pred_id == 0:
            # Don't score the background
            continue
        pred_mask = pred_label == pred_id
        these_truth_ids, truth_indices = np.unique(
            truth_label[pred_mask], return_index=True
        )[0]
        for j, truth_id in zip(truth_indices, these_truth_ids):
            if truth_id == 0:
                # Don't score the background
                continue
            truth_mask = truth_label == truth_id
            cost_matrix[j, i] = jaccard_score(truth_mask.flatten(), pred_mask.flatten())

    # Match the predicted instances to the ground truth instances
    row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)

    # Contruct the volume for the matched instances
    matched_pred_label = np.zeros_like(pred_label)
    for i, j in zip(col_inds, row_inds):
        pred_mask = pred_label == pred_ids[i]
        matched_pred_label[pred_mask] = truth_ids[j]

    hausdorff_distances = []
    for truth_id in truth_ids:
        if truth_id == 0:
            # Don't score the background
            continue
        h_dist = hausdorff_distance(
            truth_label == truth_id, matched_pred_label == truth_id
        )
        h_dist = min(h_dist, HAUSDORFF_DISTANCE_MAX)
        hausdorff_distances.append(h_dist)

    # Compute the scores
    accuracy = accuracy_score(truth_label.flatten(), matched_pred_label.flatten())
    hausdorff_distance = np.mean(hausdorff_distances)
    combined_score = (accuracy * hausdorff_distance) ** 0.5
    return {
        "accuracy": accuracy,
        "hausdorff_distance": hausdorff_distance,
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
    pred_label = (pred_label > 0).flatten()
    truth_label = (truth_label > 0).flatten()
    # Compute the scores
    scores = {
        "jaccard_score": jaccard_score(truth_label, pred_label),
        "dice_score": dice_score(truth_label, pred_label),
    }

    return scores


def score_label(pred_label_path) -> dict[str, float]:
    """
    Score a single label volume against the ground truth label volume.

    Args:
        pred_label_path (str): The path to the predicted label volume.

    Returns:
        dict: A dictionary of scores for the label volume.

    Example usage:
        scores = score_label('pred.zarr/test_volume/label1')
    """
    # Load the predicted and ground truth label volumes
    label_name = UPath(pred_label_path).name
    volume_name = UPath(pred_label_path).parent.name
    pred_label = zarr.open(pred_label_path)
    truth_label = zarr.open(UPath(TRUTH_PATH) / volume_name / label_name)
    mask_path = UPath(TRUTH_PATH) / volume_name / f"{label_name}_mask"
    if mask_path.exists():
        # Mask out uncertain regions resulting from low-res ground truth annotations
        mask = zarr.open()
        pred_label = pred_label * mask
        truth_label = truth_label * mask

    # Check if the label volumes have the same shape
    assert (
        pred_label.shape == truth_label.shape
    ), "The predicted and ground truth label volumes must have the same shape."

    # Compute the scores
    if label_name in INSTANCE_CLASSES:
        return score_instance(pred_label, truth_label)
    else:
        return score_semantic(pred_label, truth_label)


def score_volume(pred_volume_path) -> dict[str, dict[str, float]]:
    """
    Score a single volume against the ground truth volume.

    Args:
        pred_volume_path (str): The path to the predicted volume.

    Returns:
        dict: A dictionary of scores for the volume.

    Example usage:
        scores = score_volume('pred.zarr/test_volume')
    """
    # Find labels to score
    pred_labels = [a for a in zarr.open(pred_volume_path).array_keys()]

    volume_name = UPath(pred_volume_path).name
    truth_labels = [a for a in zarr.open(UPath(TRUTH_PATH) / volume_name).array_keys()]

    labels = list(set(pred_labels) & set(truth_labels))

    # Score each label
    scores = {label: score_label(pred_volume_path / label) for label in labels}

    return scores


def score_submission(
    submission_path, save_path=None
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Score a submission against the ground truth data.

    Args:
        submission_path (str): The path to the zipped submission Zarr-2 file.
        save_path (str): The path to save the scores.

    Returns:
        dict: A dictionary of scores for the submission.

    Example usage:
        scores = score_submission('submission.zip')
    """
    # Unzip the submission
    submission_path = unzip_file(submission_path)

    # Load the submission
    submission = zarr.open(submission_path)

    # Find volumes to score
    pred_volumes = [a for a in submission.array_keys()]
    truth_volumes = [a for a in zarr.open(TRUTH_PATH).array_keys()]

    volumes = list(set(pred_volumes) & set(truth_volumes))

    # Score each volume
    scores = {volume: score_volume(submission_path / volume) for volume in pred_volumes}

    # Save the scores
    if save_path:
        with open(save_path, "w") as f:
            json.dump(scores, f)
    else:
        return scores


if __name__ == "__main__":
    # Evaluate a submission
    # example usage: python evaluate.py submission.zip
    score_submission(sys.argv[1])
