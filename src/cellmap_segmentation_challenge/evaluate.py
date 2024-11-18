import argparse
import json
import os
import zipfile
from glob import glob

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import dice  # , jaccard
from skimage.measure import label as relabel
from skimage.metrics import hausdorff_distance
from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
from upath import UPath

from cellmap_data import CellMapImage

from .config import PROCESSED_PATH, SUBMISSION_PATH, REPO_ROOT

INSTANCE_CLASSES = [
    "nuc",
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

CLASS_RESOLUTIONS = {
    "nuc": 16,
    "mito": 16,
    "cell": 16,
    "default": 4,
    "eres": 2,
    "ld": 2,
}

TEST_CROPS = [234]

CROP_SHAPE = {
    234: {
        "nuc": (25, 25, 25),
        "mito": (50, 50, 50),
        "er": (200, 200, 200),
    }
}

HAUSDORFF_DISTANCE_MAX = np.inf

TRUTH_PATH = (REPO_ROOT / "data/ground_truth.zarr").path


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
    save_path, test_volume_name, label_name, labels, overwrite=False
):
    """
    Save a single 3D numpy array of class labels to a
    Zarr-2 file with the required structure.

    Args:
        save_path (str): The path to save the Zarr-2 file (ending with <filename>.zarr).
        test_volume_name (str): The name of the test volume.
        label_names (str): The names of the labels.
        labels (np.ndarray): A 3D numpy array of class labels.

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
        zarr_group[test_volume_name].create_dataset(
            label_name,
            data=(labels == i + 1),
            chunks=(64, 64, 64),
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )

    print("Done saving")


def save_numpy_class_arrays_to_zarr(
    save_path, test_volume_name, label_names, labels, overwrite=False
):
    """
    Save a list of 3D numpy arrays of binary or instance labels to a
    Zarr-2 file with the required structure.

    Args:
        save_path (str): The path to save the Zarr-2 file (ending with <filename>.zarr).
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
    if not UPath(save_path).exists():
        os.makedirs(UPath(save_path).parent, exist_ok=True)
    print(f"Saving to {save_path}")
    store = zarr.DirectoryStore(save_path)
    zarr_group = zarr.group(store)

    # Save the test volume group
    zarr_group.create_group(test_volume_name, overwrite=overwrite)

    # Save the labels
    for i, label_name in enumerate(label_names):
        print(f"Saving {label_name}")
        zarr_group[test_volume_name].create_dataset(
            label_name,
            data=labels[i],
            chunks=(64, 64, 64),
            # compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
        )

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


def score_instance(
    pred_label, truth_label, hausdorff_distance_max=HAUSDORFF_DISTANCE_MAX
) -> dict[str, float]:
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
    print("Scoring instance segmentation...")
    pred_label = relabel(pred_label, connectivity=len(pred_label.shape))
    # pred_label = label(pred_label)

    # Construct the cost matrix for Hungarian matching
    pred_ids = np.unique(pred_label)
    truth_ids = np.unique(truth_label)
    cost_matrix = np.zeros((len(truth_ids), len(pred_ids)))
    bar = tqdm(pred_ids, desc="Computing cost matrix", leave=True)
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
    for i, j in tqdm(zip(col_inds, row_inds), desc="Relabeled matched instances"):
        if pred_ids[i] == 0 or truth_ids[j] == 0:
            # Don't score the background
            continue
        pred_mask = pred_label == pred_ids[i]
        matched_pred_label[pred_mask] = truth_ids[j]

    hausdorff_distances = []
    for truth_id in tqdm(truth_ids, desc="Computing Hausdorff distances"):
        if truth_id == 0:
            # Don't score the background
            continue
        h_dist = hausdorff_distance(
            truth_label == truth_id, matched_pred_label == truth_id
        )
        h_dist = min(h_dist, hausdorff_distance_max)
        hausdorff_distances.append(h_dist)

    # Compute the scores
    accuracy = accuracy_score(truth_label.flatten(), matched_pred_label.flatten())
    hausdorff_dist = np.mean(hausdorff_distances) if hausdorff_distances else 0
    normalized_hausdorff_dist = 32 ** (
        -hausdorff_dist
    )  # normalize Hausdorff distance to [0, 1]. 32 is abritrary chosen to have a reasonable range
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
    scores = {
        "iou": jaccard_score(truth_label, pred_label),
        "dice_score": 1 - dice(truth_label, pred_label),
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
    # Load the predicted and ground truth label volumes
    label_name = UPath(pred_label_path).name
    volume_name = UPath(pred_label_path).parent.name
    pred_label = zarr.open(pred_label_path)[:]
    truth_label_path = (UPath(truth_path) / volume_name / label_name).path
    truth_label = zarr.open(truth_label_path)[:]

    # Check if the label volumes have the same shape
    if not (pred_label.shape == truth_label.shape):
        print(
            f"Shapes of {pred_label_path} and {truth_label_path} do not match: {pred_label.shape} vs {truth_label.shape}"
        )
        # Make the submission match the truth shape
        pred_label = resize_array(pred_label, truth_label.shape)

    mask_path = UPath(truth_path) / volume_name / f"{label_name}_mask"
    if mask_path.exists():
        # Mask out uncertain regions resulting from low-res ground truth annotations
        print(f"Masking {label_name} with {mask_path}...")
        mask = zarr.open(mask_path.path)[:]
        pred_label = pred_label * mask
        truth_label = truth_label * mask

    # Compute the scores
    if label_name in instance_classes:
        return score_instance(pred_label, truth_label)
    else:
        return score_semantic(pred_label, truth_label)


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

    # Find labels to score
    pred_labels = [a for a in zarr.open(pred_volume_path.path).array_keys()]

    volume_name = pred_volume_path.name
    truth_labels = [
        a for a in zarr.open((UPath(truth_path) / volume_name).path).array_keys()
    ]

    labels = list(set(pred_labels) & set(truth_labels))

    # Score each label
    scores = {
        label: score_label(
            os.path.join(pred_volume_path, label),
            truth_path=truth_path,
            instance_classes=instance_classes,
        )
        for label in labels
    }
    scores["num_voxels"] = int(
        np.prod(zarr.open((pred_volume_path / labels[0]).path).shape)
    )

    return scores


def score_submission(
    submission_path=UPath(SUBMISSION_PATH).with_suffix(".zip").path,
    result_file=None,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
) -> dict[str, dict[str, dict[str, float]]]:
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
    print(f"Volumes: {pred_volumes}")
    print(f"Truth path: {truth_path}")
    truth_volumes = [d.name for d in UPath(truth_path).glob("*") if d.is_dir()]
    print(f"Truth volumes: {truth_volumes}")

    volumes = list(set(pred_volumes) & set(truth_volumes))
    if len(volumes) == 0:
        raise ValueError(
            "No volumes found to score. Make sure the submission is formatted correctly."
        )
    print(f"Scoring volumes: {volumes}")

    # Score each volume
    scores = {
        volume: score_volume(
            submission_path / volume,
            truth_path=truth_path,
            instance_classes=instance_classes,
        )
        for volume in volumes
    }

    # Combine label scores across volumes, normalizing by the number of voxels
    print("Combining label scores...")
    label_scores = {}
    for volume in volumes:
        for label in scores[volume]:
            if label == "num_voxels":
                continue
            elif label in instance_classes:
                if label not in label_scores:
                    label_scores[label] = {
                        "accuracy": 0,
                        "hausdorff_distance": 0,
                        "normalized_hausdorff_distance": 0,
                        "combined_score": 0,
                    }
                label_scores[label]["accuracy"] += (
                    scores[volume][label]["accuracy"] / scores[volume]["num_voxels"]
                )
                label_scores[label]["hausdorff_distance"] += (
                    scores[volume][label]["hausdorff_distance"]
                    / scores[volume]["num_voxels"]
                )
                label_scores[label]["normalized_hausdorff_distance"] += (
                    scores[volume][label]["normalized_hausdorff_distance"]
                    / scores[volume]["num_voxels"]
                )
                label_scores[label]["combined_score"] += (
                    scores[volume][label]["combined_score"]
                    / scores[volume]["num_voxels"]
                )
            else:
                if label not in label_scores:
                    label_scores[label] = {"iou": 0, "dice_score": 0}
                label_scores[label]["iou"] += (
                    scores[volume][label]["iou"] / scores[volume]["num_voxels"]
                )
                label_scores[label]["dice_score"] += (
                    scores[volume][label]["dice_score"] / scores[volume]["num_voxels"]
                )
    scores["label_scores"] = label_scores

    # Compute the overall score
    print("Computing overall scores...")
    overall_instance_scores = []
    overall_semantic_scores = []
    for label in label_scores:
        if label in instance_classes:
            overall_instance_scores += [label_scores[label]["combined_score"]]
        else:
            overall_semantic_scores += [label_scores[label]["dice_score"]]
    scores["overall_instance_score"] = np.mean(overall_instance_scores)
    scores["overall_semantic_score"] = np.mean(overall_semantic_scores)
    scores["overall_score"] = (
        scores["overall_instance_score"] * scores["overall_semantic_score"]
    ) ** 0.5  # geometric mean

    # Save the scores
    if result_file:
        print(f"Saving scores to {result_file}...")
        with open(result_file, "w") as f:
            json.dump(scores, f)
    else:
        return scores


def package_submission(
    input_search_path: str | UPath = PROCESSED_PATH,
    output_path: str | UPath = SUBMISSION_PATH,
):
    """
    Package a submission for the CellMap challenge. This will create a zarr file, combining all the processed volumes, and then zip it.

    Args:
        input_search_path (str): The base path to the processed volumes, with placeholders for dataset and crops.
        output_path (str | UPath): The path to save the submission zarr to. (ending with `<filename>.zarr`; `.zarr` will be appended if not present, and replaced with `.zip` when zipped).
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
        crop_group = zarr_group.create_group(f"crop{crop}")
        crop_path = input_search_path.format(dataset="*", crop=f"crop{crop}")
        crop_path = glob(crop_path)
        if len(crop_path) == 0:
            print(f"Skipping {crop_path} as it does not exist.")
            continue
        crop_path = crop_path[0]

        # Find all the processed labels for the test volume
        test_volume = UPath(crop_path).name
        labels = [d.name for d in UPath(crop_path).glob("*") if d.is_dir()]
        print(f"Found labels for {test_volume}: {labels}")

        # Rescale the processed volumes to match the expected submission resolution if required
        for label in labels:
            if label in CLASS_RESOLUTIONS:
                resolution = CLASS_RESOLUTIONS[label]
            else:
                resolution = CLASS_RESOLUTIONS["default"]
            label_array = crop_group.create_dataset(
                label,
                overwrite=True,
                shape=CROP_SHAPE[crop][label],
            )
            print(f"Scaling {label} to {resolution}nm")
            image = CellMapImage(
                path=(UPath(crop_path) / label).path,
                target_class=label,
                target_scale=[
                    resolution,
                ]
                * 3,
                target_voxel_shape=CROP_SHAPE[crop][label],
                pad=True,
                pad_value=0,
            )
            # Save the processed labels to the submission zarr
            label_array[:] = image[image.center].cpu().numpy()

    print(f"Saved submission to {output_path}")

    print("Zipping submission...")
    zip_submission(output_path)

    print("Done packaging submission")


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
