import argparse
import json
import os
from time import time, sleep
import zipfile

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import dice
from skimage.measure import label as relabel
from skimage.transform import rescale

from pykdtree.kdtree import KDTree as cKDTree

from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
from upath import UPath

from cellmap_data import CellMapImage

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .config import PROCESSED_PATH, SUBMISSION_PATH, TRUTH_PATH
from .utils import TEST_CROPS_DICT

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
CAST_TO_NONE = [np.nan, np.inf, -np.inf]

MAX_INSTANCE_THREADS = int(os.getenv("MAX_INSTANCE_THREADS", 1))
MAX_SEMANTIC_THREADS = int(os.getenv("MAX_SEMANTIC_THREADS", 20))
PER_INSTANCE_THREADS = int(os.getenv("PER_INSTANCE_THREADS", 16))
# submitted_# of instances / ground_truth_# of instances
INSTANCE_RATIO_CUTOFF = float(os.getenv("INSTANCE_RATIO_CUTOFF", 50))
PRECOMPUTE_LIMIT = int(os.getenv("PRECOMPUTE_LIMIT", 1e7))
DEBUG = os.getenv("DEBUG", "False") != "False"


class spoof_precomputed:
    def __init__(self, array, ids):
        self.array = array
        self.ids = ids
        self.index = -1

    def __getitem__(self, ids):
        if isinstance(ids, int):
            return np.array(self.array == self.ids[ids], dtype=bool)
        return np.array([self.array == self.ids[i] for i in ids], dtype=bool)

    def __len__(self):
        return len(self.ids)


def optimized_hausdorff_distances(
    truth_label,
    matched_pred_label,
    voxel_size,
    hausdorff_distance_max,
    method="standard",
):
    # Get unique truth IDs, excluding the background (0)
    truth_ids = np.unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]  # Exclude background
    if len(truth_ids) == 0:
        return []

    def get_distance(i):
        # Skip if both masks are empty
        truth_mask = truth_label == truth_ids[i]
        pred_mask = matched_pred_label == truth_ids[i]
        if not np.any(truth_mask) and not np.any(pred_mask):
            return 0

        # Compute Hausdorff distance for the current pair
        h_dist = compute_hausdorff_distance(
            truth_mask,
            pred_mask,
            voxel_size,
            hausdorff_distance_max,
            method,
        )
        return i, h_dist

    # Initialize list for distances
    hausdorff_distances = np.empty(len(truth_ids))
    if DEBUG:
        # Use tqdm for progress tracking
        bar = tqdm(
            range(len(truth_ids)),
            desc="Computing Hausdorff distances",
            leave=True,
            dynamic_ncols=True,
            total=len(truth_ids),
        )
        # Compute the cost matrix
        for i in bar:
            i, h_dist = get_distance(i)
            hausdorff_distances[i] = h_dist
    else:
        with ThreadPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
            for i, h_dist in tqdm(
                executor.map(get_distance, range(len(truth_ids))),
                desc="Computing Hausdorff distances",
                total=len(truth_ids),
                dynamic_ncols=True,
            ):
                hausdorff_distances[i] = h_dist

    return hausdorff_distances


def compute_hausdorff_distance(image0, image1, voxel_size, max_distance, method):
    """
    Compute the Hausdorff distance between two binary masks, optimized for pre-vectorized inputs.
    """
    # Extract nonzero points
    a_points = np.argwhere(image0)
    b_points = np.argwhere(image1)

    # Handle empty sets
    if len(a_points) == 0 and len(b_points) == 0:
        return 0
    elif len(a_points) == 0 or len(b_points) == 0:
        return np.inf

    # Scale points by voxel size
    a_points = a_points * np.array(voxel_size)
    b_points = b_points * np.array(voxel_size)

    # Build KD-trees once
    a_tree = cKDTree(a_points)
    b_tree = cKDTree(b_points)

    # Query distances
    # fwd = a_tree.query(b_points, k=1, distance_upper_bound=max_distance)[0]
    # bwd = b_tree.query(a_points, k=1, distance_upper_bound=max_distance)[0]
    fwd = a_tree.query(b_points, k=1)[0]
    bwd = b_tree.query(a_points, k=1)[0]

    # Replace "inf" with `max_distance` for numerical stability
    # fwd[fwd == np.inf] = max_distance
    # bwd[bwd == np.inf] = max_distance
    fwd[fwd > max_distance] = max_distance
    bwd[bwd > max_distance] = max_distance

    if method == "standard":
        return max(fwd.max(), bwd.max())
    elif method == "modified":
        return max(fwd.mean(), bwd.mean())


def score_instance(
    pred_label,
    truth_label,
    voxel_size,
    hausdorff_distance_max=HAUSDORFF_DISTANCE_MAX,
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
    logging.info("Scoring instance segmentation...")
    # Relabel the predicted instance labels to be consistent with the ground truth instance labels
    logging.info("Relabeling predicted instance labels...")
    pred_label = relabel(pred_label, connectivity=len(pred_label.shape))

    # Get unique IDs, excluding background (assumed to be 0)
    truth_ids = np.unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]

    pred_ids = np.unique(pred_label)
    pred_ids = pred_ids[pred_ids != 0]

    # Skip if the submission has way too many instances
    if len(truth_ids) > 0 and len(pred_ids) / len(truth_ids) > INSTANCE_RATIO_CUTOFF:
        logging.warning(
            f"WARNING: Skipping {len(pred_ids)} instances in submission, {len(truth_ids)} in ground truth, because there are too many instances in the submission."
        )
        return {
            "accuracy": 0,
            "hausdorff_distance": np.inf,
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
        }

    # Flatten the labels for vectorized computation
    truth_flat = truth_label.flatten()
    pred_flat = pred_label.flatten()

    matched_pred_label = np.zeros_like(pred_label)

    if len(pred_ids) > 0:

        # Precompute binary masks for all `truth_ids`
        if len(truth_flat) * len(truth_ids) > PRECOMPUTE_LIMIT:
            truth_binary_masks = spoof_precomputed(truth_flat, truth_ids)
        else:
            logging.info("Precomputing binary masks for all `truth_ids`...")
            truth_binary_masks = np.array(
                [(truth_flat == tid) for tid in truth_ids], dtype=bool
            )

        def get_cost(j):
            # Find all `truth_ids` that overlap with this prediction mask
            pred_mask = pred_flat == pred_ids[j]
            relevant_truth_ids = np.unique(truth_flat[pred_mask])
            relevant_truth_ids = relevant_truth_ids[relevant_truth_ids != 0]
            relevant_truth_indices = np.where(np.isin(truth_ids, relevant_truth_ids))[0]
            relevant_truth_masks = truth_binary_masks[relevant_truth_indices]

            if relevant_truth_indices.size == 0:
                return [], j, []

            tp = relevant_truth_masks[:, pred_mask].sum(1)
            fn = (relevant_truth_masks[:, pred_mask == 0]).sum(1)
            fp = (relevant_truth_masks[:, pred_mask] == 0).sum(1)

            # Compute Jaccard scores
            jaccard_scores = tp / (tp + fp + fn)

            # Fill in the cost matrix for this `j` (prediction)
            return relevant_truth_indices, j, jaccard_scores

        # Initialize the cost matrix
        logging.info(
            f"Initializing cost matrix of {len(truth_ids)} x {len(pred_ids)} (true x pred)..."
        )
        cost_matrix = np.zeros((len(truth_ids), len(pred_ids)))

        # Compute the cost matrix
        if DEBUG:
            # Use tqdm for progress tracking
            bar = tqdm(
                range(pred_ids),
                desc="Computing cost matrix",
                leave=True,
                dynamic_ncols=True,
                total=len(pred_ids),
            )
            # Compute the cost matrix
            for j in bar:
                relevant_truth_indices, j, jaccard_scores = get_cost(j)
                cost_matrix[relevant_truth_indices, j] = jaccard_scores
        else:
            with ThreadPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
                for relevant_truth_indices, j, jaccard_scores in tqdm(
                    executor.map(get_cost, range(len(pred_ids))),
                    desc="Computing cost matrix in parallel",
                    dynamic_ncols=True,
                    total=len(pred_ids),
                    leave=True,
                ):
                    cost_matrix[relevant_truth_indices, j] = jaccard_scores

        # Match the predicted instances to the ground truth instances
        logging.info("Calculating linear sum assignment...")
        row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)

        # Contruct the volume for the matched instances
        for i, j in tqdm(
            zip(col_inds, row_inds),
            desc="Relabeling matched instances",
            dynamic_ncols=True,
        ):
            if pred_ids[i] == 0 or truth_ids[j] == 0:
                # Don't score the background
                continue
            pred_mask = pred_label == pred_ids[i]
            matched_pred_label[pred_mask] = truth_ids[j]

        hausdorff_distances = optimized_hausdorff_distances(
            truth_label, matched_pred_label, voxel_size, hausdorff_distance_max
        )
    else:
        # No predictions to match
        hausdorff_distances = []

    # Compute the scores
    logging.info("Computing accuracy score...")
    accuracy = accuracy_score(truth_flat, matched_pred_label.flatten())
    hausdorff_dist = np.mean(hausdorff_distances) if len(hausdorff_distances) > 0 else 0
    normalized_hausdorff_dist = 1.01 ** (
        -hausdorff_dist / np.linalg.norm(voxel_size)
    )  # normalize Hausdorff distance to [0, 1] using the maximum distance represented by a voxel. 32 is arbitrarily chosen to have a reasonable range
    combined_score = (accuracy * normalized_hausdorff_dist) ** 0.5
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    logging.info(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    logging.info(f"Combined Score: {combined_score:.4f}")
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
    logging.info("Scoring semantic segmentation...")
    # Flatten the label volumes and convert to binary
    pred_label = (pred_label > 0.0).flatten()
    truth_label = (truth_label > 0.0).flatten()
    # Compute the scores

    if np.sum(truth_label + pred_label) == 0:
        # If there are no true positives, set the scores to 1
        logging.debug("No true positives found. Setting scores to 1.")
        dice_score = 1
    else:
        dice_score = 1 - dice(truth_label, pred_label)
    scores = {
        "iou": jaccard_score(truth_label, pred_label, zero_division=1),
        "dice_score": dice_score if not np.isnan(dice_score) else 1,
    }

    logging.info(f"IoU: {scores['iou']:.4f}")
    logging.info(f"Dice Score: {scores['dice_score']:.4f}")

    return scores


def score_label(
    pred_label_path,
    label_name,
    crop_name,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
) -> dict[str, float]:
    """
    Score a single label volume against the ground truth label volume.

    Args:
        pred_label_path (str): The path to the predicted label volume.
        truth_path (str): The path to the ground truth label volume.
        instance_classes (list): A list of instance classes.

    Returns:
        dict: A dictionary of scores for the label volume.

    Example usage:
        scores = score_label('pred.zarr/test_volume/label1')
    """
    try:
        if pred_label_path is None:
            logging.info(
                f"Label {label_name} not found in submission volume {crop_name}."
            )
            return (
                crop_name,
                label_name,
                empty_label_score(
                    label=label_name,
                    crop_name=crop_name,
                    instance_classes=instance_classes,
                    truth_path=truth_path,
                ),
            )
        logging.info(f"Scoring {crop_name}/{label_name}...")
        truth_path = UPath(truth_path)
        # Load the predicted and ground truth label volumes
        truth_label_path = (truth_path / crop_name / label_name).path
        try:
            truth_label_ds = zarr.open(truth_label_path, mode="r")
            truth_label = truth_label_ds[:]
        except Exception:
            raise ValueError(
                f"Failed to load ground truth data for {crop_name}/{label_name}. Please contact the challenge organizers."
            )

        crop = TEST_CROPS_DICT[int(crop_name.removeprefix("crop")), label_name]
        try:
            pred_label = match_crop_space(
                pred_label_path,
                label_name,
                crop.voxel_size,
                crop.shape,
                crop.translation,
            )
        except Exception:
            raise ValueError(
                f"Failed to process submission data for {crop_name}/{label_name}. Please verify your data format and coordinate transformations are correct."
            )
    except Exception:
        raise Exception(
            "An unexpected error occurred during label scoring. Please check your submission and contact the challenge organizers if the issue persists."
        )

    mask_path = truth_path / crop_name / f"{label_name}_mask"
    if mask_path.exists():
        # Mask out uncertain regions resulting from low-res ground truth annotations
        logging.info(f"Masking {label_name} with {mask_path}...")
        try:
            mask = zarr.open(mask_path.path, mode="r")[:]
            pred_label = pred_label * mask
            truth_label = truth_label * mask
        except Exception:
            raise ValueError(
                f"Failed to apply mask for {crop_name}/{label_name}. Please contact the challenge organizers."
            )

        # Compute the scores
    if label_name in instance_classes:
        logging.info(
            f"Starting an instance evaluation for {label_name} in {crop_name}..."
        )
        timer = time()
        try:
            results = score_instance(pred_label, truth_label, crop.voxel_size)
        except Exception:
            raise ValueError(
                f"Failed to compute instance scores for {crop_name}/{label_name}. Ensure your instance segmentation data has properly labeled instances with integer IDs."
            )
        logging.info(
            f"Finished instance evaluation for {label_name} in {crop_name} in {time() - timer:.2f} seconds..."
        )
    else:
        try:
            results = score_semantic(pred_label, truth_label)
        except Exception:
            raise ValueError(
                f"Failed to compute semantic scores for {crop_name}/{label_name}. Ensure your data contains valid probability or binary values."
            )
    results["num_voxels"] = int(np.prod(truth_label.shape))
    results["voxel_size"] = crop.voxel_size
    results["is_missing"] = False
    return crop_name, label_name, results


def empty_label_score(
    label, crop_name, instance_classes=INSTANCE_CLASSES, truth_path=TRUTH_PATH
):
    if label in instance_classes:
        return {
            "accuracy": 0,
            "hausdorff_distance": 0,
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
            "num_voxels": int(
                np.prod(
                    zarr.open((truth_path / crop_name / label).path, mode="r").shape
                )
            ),
            "voxel_size": zarr.open(
                (truth_path / crop_name / label).path, mode="r"
            ).attrs["voxel_size"],
            "is_missing": True,
        }
    else:
        return {
            "iou": 0,
            "dice_score": 0,
            "num_voxels": int(
                np.prod(
                    zarr.open((truth_path / crop_name / label).path, mode="r").shape
                )
            ),
            "voxel_size": zarr.open(
                (truth_path / crop_name / label).path, mode="r"
            ).attrs["voxel_size"],
            "is_missing": True,
        }


def get_evaluation_args(
    volumes,
    submission_path,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
) -> dict[str, dict[str, float]]:
    """
    Get the arguments for scoring each label in the submission.
    Args:
        volumes (list): A list of volumes to score.
        submission_path (str): The path to the submission volume.
        truth_path (str): The path to the ground truth volume.
        instance_classes (list): A list of instance classes.
    Returns:
        A list of tuples containing the arguments for each label to be scored.
    """
    if not isinstance(volumes, (tuple, list)):
        volumes = [volumes]
    score_label_arglist = []
    for volume in volumes:
        submission_path = UPath(submission_path)
        pred_volume_path = submission_path / volume
        logging.info(f"Scoring {pred_volume_path}...")
        truth_path = UPath(truth_path)

        # Find labels to score
        pred_labels = [
            a for a in zarr.open(pred_volume_path.path, mode="r").array_keys()
        ]

        crop_name = pred_volume_path.name
        truth_labels = [
            a for a in zarr.open((truth_path / crop_name).path, mode="r").array_keys()
        ]

        found_labels = list(set(pred_labels) & set(truth_labels))
        missing_labels = list(set(truth_labels) - set(pred_labels))

        # Score_label arguments for each label
        score_label_arglist.extend(
            [
                (
                    pred_volume_path / label if label in found_labels else None,
                    label,
                    crop_name,
                    truth_path,
                    instance_classes,
                )
                for label in truth_labels
            ]
        )
        logging.info(f"Missing labels: {missing_labels}")

    return score_label_arglist


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
    logging.info(f"Scoring missing volume {truth_volume_path}...")
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


def combine_scores(
    scores,
    include_missing=True,
    instance_classes=INSTANCE_CLASSES,
    cast_to_none=CAST_TO_NONE,
):
    """
    Combine scores across volumes, normalizing by the number of voxels.

    Args:
        scores (dict): A dictionary of scores for each volume, as returned by `score_volume`.
        include_missing (bool): Whether to include missing volumes in the combined scores.
        instance_classes (list): A list of instance classes.
        cast_to_none (list): A list of values to cast to None in the combined scores.

    Returns:
        dict: A dictionary of combined scores across all volumes.

    Example usage:
        combined_scores = combine_scores(scores)
    """

    # Combine label scores across volumes, normalizing by the number of voxels
    logging.info(f"Combining label scores...")
    scores = scores.copy()
    label_scores = {}
    total_volumes = {}
    for ds, these_scores in scores.items():
        for label, this_score in these_scores.items():
            # logging.info(this_score)
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
            else:
                if label not in label_scores:
                    label_scores[label] = {"iou": 0, "dice_score": 0}
                    total_volumes[label] = 0
            for key in label_scores[label].keys():
                if this_score[key] is None:
                    continue
                label_scores[label][key] += this_score[key] * total_volume
                if this_score[key] in cast_to_none:
                    scores[ds][label][key] = None
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
        # Cast to None if the value is in `cast_to_none`
        for key in label_scores[label]:
            if label_scores[label][key] in cast_to_none:
                label_scores[label][key] = None
    scores["label_scores"] = label_scores

    # Compute the overall score
    logging.info("Computing overall scores...")
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

    logging.info(f"Scoring {submission_path}...")
    start_time = time()
    # Unzip the submission
    try:
        submission_path = unzip_file(submission_path)
    except Exception:
        raise ValueError(
            "Failed to process submission file. Please ensure you submitted a valid .zip file containing a Zarr structure."
        )

    # Find volumes to score
    logging.info(f"Scoring volumes in {submission_path}...")
    try:
        pred_volumes = [d.name for d in UPath(submission_path).glob("*") if d.is_dir()]
        truth_path = UPath(truth_path)
        logging.info(f"Volumes: {pred_volumes}")
        logging.info(f"Truth path: {truth_path}")
        truth_volumes = [d.name for d in truth_path.glob("*") if d.is_dir()]
        logging.info(f"Truth volumes: {truth_volumes}")
    except Exception:
        raise ValueError(
            "Failed to read submission structure. Ensure your submission contains crop folders (e.g., crop557, crop558, etc.) at the top level."
        )

    found_volumes = list(set(pred_volumes) & set(truth_volumes))
    missing_volumes = list(set(truth_volumes) - set(pred_volumes))
    if len(found_volumes) == 0:
        raise ValueError(
            f"No valid test volumes found in submission. Expected volumes like: {', '.join(truth_volumes[:5])}. Please ensure your submission structure matches the required format."
        )
    logging.info(f"Scoring volumes: {found_volumes}")
    if len(missing_volumes) > 0:
        logging.info(f"Missing volumes: {missing_volumes}")
        logging.info("Scoring missing volumes as 0's")

    scores = {
        volume: missing_volume_score(
            truth_path / volume, instance_classes=instance_classes
        )
        for volume in missing_volumes
    }

    # Get all prediction paths to evaluate
    evaluation_args = get_evaluation_args(
        found_volumes,
        submission_path=UPath(submission_path),
        truth_path=truth_path,
        instance_classes=instance_classes,
    )

    # Score each volume
    if DEBUG:
        logging.info("Scoring volumes in serial for debugging...")
        results = [score_label(*args) for args in evaluation_args]
        # Combine the results into a dictionary
        for crop_name, label_name, result in results:
            if crop_name not in scores:
                scores[crop_name] = {}
            scores[crop_name][label_name] = result

        # Combine label scores across volumes, normalizing by the number of voxels
        all_scores = combine_scores(
            scores, include_missing=True, instance_classes=instance_classes
        )
        found_scores = combine_scores(
            scores, include_missing=False, instance_classes=instance_classes
        )

        # Save the scores
        if result_file:
            logging.info(f"Saving collected scores to {result_file}...")
            with open(result_file, "w") as f:
                json.dump(all_scores, f, indent=4)

            found_result_file = str(result_file).replace(
                UPath(result_file).suffix, "_submitted_only" + UPath(result_file).suffix
            )
            logging.info(
                f"Saving scores for only submitted data to {found_result_file}..."
            )
            with open(found_result_file, "w") as f:
                json.dump(found_scores, f, indent=4)
            logging.info(
                f"Scores saved to {result_file} and {found_result_file} in {time() - start_time:.2f} seconds"
            )
        else:
            return all_scores
    else:
        logging.info("Scoring volumes in parallel...")
        instance_pool = ProcessPoolExecutor(MAX_INSTANCE_THREADS)
        semantic_pool = ProcessPoolExecutor(MAX_SEMANTIC_THREADS)
        futures = []
        for args in evaluation_args:
            if args[1] in instance_classes:
                futures.append(instance_pool.submit(score_label, *args))
            else:
                futures.append(semantic_pool.submit(score_label, *args))
        results = []
        for future in tqdm(
            as_completed(futures),
            desc="Scoring volumes",
            total=len(futures),
            dynamic_ncols=True,
            leave=True,
        ):
            results.append(future.result())
            all_scores, found_scores = update_scores(
                scores, results, result_file, instance_classes=instance_classes
            )

    logging.info("Scores combined across all test volumes:")
    logging.info(
        f"\tOverall Instance Score: {all_scores['overall_instance_score']:.4f}"
    )
    logging.info(
        f"\tOverall Semantic Score: {all_scores['overall_semantic_score']:.4f}"
    )
    logging.info(f"\tOverall Score: {all_scores['overall_score']:.4f}")

    logging.info("Scores combined across test volumes with data submitted:")
    logging.info(
        f"\tOverall Instance Score: {found_scores['overall_instance_score']:.4f}"
    )
    logging.info(
        f"\tOverall Semantic Score: {found_scores['overall_semantic_score']:.4f}"
    )
    logging.info(f"\tOverall Score: {found_scores['overall_score']:.4f}")
    logging.info(f"Submission scored in {time() - start_time:.2f} seconds")

    if result_file is None:
        return all_scores


def num_evals_done(all_scores):
    num_evals_done = 0
    for volume, scores in all_scores.items():
        if "crop" in volume:
            num_evals_done += len(scores.keys())
    return num_evals_done


def update_scores(scores, results, result_file, instance_classes=INSTANCE_CLASSES):
    start_time = time()
    logging.info(f"Updating scores in {result_file}...")

    # Check the types of the inputs
    assert isinstance(scores, dict)
    assert isinstance(results, list)

    # Combine the results into a dictionary
    # TODO: This is technically inefficient, but it works for now
    for crop_name, label_name, result in results:
        if crop_name not in scores:
            scores[crop_name] = {}
        scores[crop_name][label_name] = result

    # Combine label scores across volumes, normalizing by the number of voxels
    all_scores = combine_scores(
        scores, include_missing=True, instance_classes=instance_classes
    )
    all_scores["total_evals"] = len(TEST_CROPS_DICT)
    all_scores["num_evals_done"] = num_evals_done(all_scores)

    found_scores = combine_scores(
        scores, include_missing=False, instance_classes=instance_classes
    )

    if result_file is not None:
        with open(result_file, "w") as f:
            json.dump(all_scores, f, indent=4)

        found_result_file = str(result_file).replace(
            UPath(result_file).suffix, "_submitted_only" + UPath(result_file).suffix
        )
        with open(found_result_file, "w") as f:
            json.dump(found_scores, f, indent=4)

        logging.info(
            f"Scores updated in {result_file} and {found_result_file} in {time() - start_time:.2f} seconds"
        )

    return all_scores, found_scores


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
    try:
        ds = zarr.open(str(path), mode="r")
    except Exception:
        raise ValueError(
            f"Cannot open zarr array at path: {UPath(path).name}. Ensure your submission is a valid Zarr format."
        )
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
        logging.info(f"Could not find voxel size and translation for {path}")
        logging.info(
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
        logging.info(
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
                logging.info(
                    "WARNING: Cropping to proper offset resulted in empty volume."
                )
                logging.info(f"\tInput shape: {image.shape}, Output shape: {shape}")
                logging.info(
                    f"\tInput offset: {input_start}, Output offset: {output_start}"
                )
                return result

            input_slices.append(slice(int(input_start), int(input_end)))
            output_slices.append(slice(int(output_start), int(output_end)))

        # Copy the data
        result[*output_slices] = image[*input_slices]
        return result
    else:
        return image


def unzip_file(zip_path):
    """
    Unzip a zip file to a specified directory.

    Args:
        zip_path (str): The path to the zip file.

    Example usage:
        unzip_file('submission.zip')
    """
    try:
        saved_path = UPath(zip_path).with_suffix(".zarr").path
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(saved_path)
            logging.info(f"Unzipped {zip_path} to {saved_path}")
        return UPath(saved_path)
    except zipfile.BadZipFile:
        raise ValueError(
            f"Invalid zip file. Please ensure you submitted a valid .zip file."
        )
    except Exception:
        raise ValueError(
            f"Failed to extract submission file. Please verify the file is not corrupted."
        )


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
