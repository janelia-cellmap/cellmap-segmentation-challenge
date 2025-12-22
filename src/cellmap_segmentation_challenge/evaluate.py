import argparse
import json
import os
from time import time
import zipfile

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import dice
from fastremap import remap, unique
import cc3d
from skimage.transform import rescale

from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
from upath import UPath

from cellmap_data import CellMapImage

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .config import SUBMISSION_PATH, TRUTH_PATH, INSTANCE_CLASSES
from .utils import TEST_CROPS_DICT

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

HAUSDORFF_DISTANCE_MAX = np.inf
CAST_TO_NONE = [np.nan, np.inf, -np.inf]

MAX_INSTANCE_THREADS = int(os.getenv("MAX_INSTANCE_THREADS", 1))
MAX_SEMANTIC_THREADS = int(os.getenv("MAX_SEMANTIC_THREADS", 20))
PER_INSTANCE_THREADS = int(os.getenv("PER_INSTANCE_THREADS", 16))
# submitted_# of instances / ground_truth_# of instances
PRECOMPUTE_LIMIT = int(os.getenv("PRECOMPUTE_LIMIT", 1e7))
DEBUG = os.getenv("DEBUG", "False").lower() != "false"


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray | None:
    """
    Compute IoU between all GT and Pred instance IDs.
    Assumes IDs range from 1 to max(ID) (0 is background). If IDs are non-sequential (e.g., 1, 2, 5), the output matrix will contain empty rows/columns for missing IDs.
    Returns float32 array of shape (max(gt) + 1, max(pred) + 1), where rows/columns for missing IDs will be empty.
    """
    INSTANCE_RATIO_CUTOFF = float(os.getenv("INSTANCE_RATIO_CUTOFF", 50))

    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have the same shape")

    # 1D views without copying if possible
    g = np.ravel(gt)
    p = np.ravel(pred)

    # Number of instances (sequential ids -> max id)
    nG = int(g.max()) if g.size else 0
    nP = int(p.max()) if p.size else 0

    # Early exits
    if nG == 0 or nP == 0:
        if nG == 0 and nP > 0:
            logging.info("No GT instances; returning empty IoU with pred columns.")
        if nP == 0 and nG > 0:
            logging.info("No Pred instances; returning empty IoU with gt rows.")
        return np.zeros((nG, nP), dtype=np.float32)

    if (nP / nG) > INSTANCE_RATIO_CUTOFF:
        logging.warning(
            f"WARNING: Skipping {nP} instances in submission, {nG} in ground truth, "
            f"because there are too many instances in the submission."
        )
        return None

    # Foreground (non-background) mask for each side and for pairwise overlaps
    g_fg = g > 0
    p_fg = p > 0
    fg = g_fg & p_fg

    # ---- Per-object areas (sizes) ----
    # Use uint32 where possible to reduce memory; cast to int64 for safety if needed.
    gt_sizes = np.bincount((g[g_fg].astype(np.int64) - 1), minlength=nG)[:, None]
    pr_sizes = np.bincount((p[p_fg].astype(np.int64) - 1), minlength=nP)[None, :]

    # ---- Intersections for observed pairs only (sparse counting) ----
    gi = g[fg].astype(np.int64) - 1
    pj = p[fg].astype(np.int64) - 1
    if gi.size == 0:
        # No overlaps anywhere -> IoU is all zeros
        return np.zeros((nG, nP), dtype=np.float32)

    # Encode pairs to a single 64-bit key and count only present pairs
    # Use unsigned to avoid negative-overflow corner cases.
    gi_u = gi.astype(np.uint64)
    pj_u = pj.astype(np.uint64)
    key = gi_u * np.uint64(nP) + pj_u

    uniq_keys, counts = np.unique(key, return_counts=True)
    rows = (uniq_keys // np.uint64(nP)).astype(np.int64)
    cols = (uniq_keys % np.uint64(nP)).astype(np.int64)

    # ---- IoU only for observed pairs, then scatter into dense matrix ----
    # union_ij = gt_sizes[i] + pr_sizes[j] - inter_ij
    inter_ij = counts.astype(np.int64)
    union_ij = gt_sizes[rows, 0] + pr_sizes[0, cols] - inter_ij
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_vals = (inter_ij / union_ij).astype(np.float32)

    iou = np.zeros((nG, nP), dtype=np.float32)
    iou[rows, cols] = iou_vals  # all other entries remain 0

    return iou


def optimized_hausdorff_distances(
    truth_label,
    pred_label,
    voxel_size,
    hausdorff_distance_max,
    method="standard",
    percentile: float | None = None,
):
    """
    Compute per-truth-instance Hausdorff-like distances against the (already remapped)
    prediction using multithreading. Returns a 1D float32 numpy array whose i-th
    entry corresponds to truth_ids[i].

    Parameters
    ----------
    truth_label : np.ndarray
        Ground-truth instance label volume (0 == background).
    pred_label : np.ndarray
        Prediction instance label volume that has already been remapped to align
        with the GT ids (0 == background).
    voxel_size : Sequence[float]
        Physical voxel sizes in Z, Y, X (or Y, X) order.
    hausdorff_distance_max : float
        Cap for distances (use np.inf for uncapped).
    method : {"standard", "modified", "percentile"}
        "standard" -> classic Hausdorff (max of directed maxima)
        "modified" -> mean of directed distances, then max of the two means
        "percentile" -> use the given percentile of directed distances (requires
                         `percentile` to be provided).
    percentile : float | None
        Percentile (0-100) used when method=="percentile".
    """
    # Unique GT ids (exclude background = 0)
    truth_ids = unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]
    true_num = int(truth_ids.size)
    if true_num == 0:
        return np.empty((0,), dtype=np.float32)

    voxel_size = np.asarray(voxel_size, dtype=np.float64)

    def get_distance(i: int):
        tid = int(truth_ids[i])
        truth_mask = truth_label == tid
        pred_mask = pred_label == tid
        # Note: because tid comes from truth_label, truth_mask has at least one voxel
        # Compute directed/undirected Hausdorff according to method
        h_dist = compute_hausdorff_distance(
            truth_mask,
            pred_mask,
            voxel_size,
            hausdorff_distance_max,
            method=method,
            percentile=percentile,
        )
        return i, float(h_dist)

    dists = np.empty((true_num,), dtype=np.float32)

    if DEBUG:
        for i in tqdm(
            range(true_num),
            desc="Computing Hausdorff distances",
            leave=True,
            dynamic_ncols=True,
            total=true_num,
        ):
            idx, h = get_distance(i)
            dists[idx] = h
    else:
        with ThreadPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
            for idx, h in tqdm(
                executor.map(get_distance, range(true_num)),
                desc="Computing Hausdorff distances",
                total=true_num,
                dynamic_ncols=True,
            ):
                dists[idx] = h

    return dists


def compute_hausdorff_distance(
    image0,
    image1,
    voxel_size,
    max_distance: float,
    method: str = "standard",
    percentile: float | None = None,
):
    """
    Compute the (undirected) Hausdorff-like distance between two binary masks using
    Euclidean distance transforms (EDT), which is generally faster and more memory
    friendly than building KD-trees for large 3D volumes.

    Parameters
    ----------
    image0, image1 : np.ndarray (bool or int)
        Binary masks (True/1 = foreground). They should already be aligned to the
        same voxel grid.
    voxel_size : Sequence[float]
        Physical voxel sizes in Z, Y, X (or Y, X) order; passed to EDT via
        `sampling` to support anisotropy.
    max_distance : float
        Distances are clipped to this value (use np.inf for no clipping).
    method : {"standard", "modified", "percentile"}
        "standard": classic Hausdorff -> max(max(dist(A→B)), max(dist(B→A)))
        "modified": robust mean -> max(mean(dist(A→B)), mean(dist(B→A)))
        "percentile": robust percentile -> max(P(dist(A→B)), P(dist(B→A))) where P
                       is the given percentile.
    percentile : float | None
        Percentile in [0, 100]; required if method == "percentile".
    """
    from scipy.ndimage import distance_transform_edt

    a = np.asarray(image0, dtype=bool)
    b = np.asarray(image1, dtype=bool)

    # Handle empty sets
    a_n = int(a.sum())
    b_n = int(b.sum())
    if a_n == 0 and b_n == 0:
        return 0.0
    if a_n == 0 or b_n == 0:
        return float(max_distance)

    voxel_size = np.asarray(voxel_size, dtype=np.float64)

    # Directed distances via EDT to the *other* set's foreground
    # distance_transform_edt computes distance to nearest zero -> pass ~mask so that
    # zeros are at foreground voxels of the other set.
    dist_to_b = distance_transform_edt(~b, sampling=voxel_size)
    dist_to_a = distance_transform_edt(~a, sampling=voxel_size)

    fwd = dist_to_b[a]
    bwd = dist_to_a[b]

    if method == "standard":
        d = max(fwd.max(initial=0.0), bwd.max(initial=0.0))
    elif method == "modified":
        d = max(fwd.mean() if fwd.size else 0.0, bwd.mean() if bwd.size else 0.0)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("'percentile' must be provided when method='percentile'")
        d = max(
            float(np.percentile(fwd, percentile)) if fwd.size else 0.0,
            float(np.percentile(bwd, percentile)) if bwd.size else 0.0,
        )
    else:
        raise ValueError("method must be one of {'standard', 'modified', 'percentile'}")

    return float(min(d, max_distance))


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
    pred_label = cc3d.connected_components(pred_label)

    # Compute the IoU cost matrix between the predicted and ground truth instance labels
    cost_matrix = iou_matrix(truth_label, pred_label)

    if cost_matrix is None:
        # Too many instances in submission, skip scoring
        return {
            "accuracy": 0,
            "hausdorff_distance": np.inf,
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
        }
    elif cost_matrix.size > 0:
        # Match the predicted instances to the ground truth instances
        logging.info("Calculating linear sum assignment...")
        row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)

        # Construct the volume for the matched instances
        mapping = {0: 0}  # background maps to background
        mapping.update(
            {pred_id + 1: truth_id + 1 for truth_id, pred_id in zip(row_inds, col_inds)}
        )
        pred_label = remap(
            pred_label, mapping, in_place=True, preserve_missing_labels=True
        )

        hausdorff_distances = optimized_hausdorff_distances(
            truth_label, pred_label, voxel_size, hausdorff_distance_max
        )
    else:
        # No predictions to match
        hausdorff_distances = []

    # Compute the scores
    logging.info("Computing accuracy score...")
    accuracy = accuracy_score(truth_label.flatten(), pred_label.flatten())
    hausdorff_dist = np.mean(hausdorff_distances) if len(hausdorff_distances) > 0 else 0
    normalized_hausdorff_dist = 1.01 ** (
        -hausdorff_dist / np.linalg.norm(voxel_size)
    )  # normalize Hausdorff distance to [0, 1] using the maximum distance represented by a voxel
    combined_score = (accuracy * normalized_hausdorff_dist) ** 0.5  # geometric mean
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    logging.info(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    logging.info(f"Combined Score: {combined_score:.4f}")
    return {
        "accuracy": accuracy,
        "hausdorff_distance": hausdorff_dist,
        "normalized_hausdorff_distance": normalized_hausdorff_dist,
        "combined_score": combined_score,
    }  # type: ignore


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
        iou_score = 1
    else:
        dice_score = 1 - dice(truth_label, pred_label)
        iou_score = jaccard_score(truth_label, pred_label, zero_division=1)
    scores = {
        "iou": iou_score,
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
    if pred_label_path is None:
        logging.info(f"Label {label_name} not found in submission volume {crop_name}.")
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
    logging.info(f"Scoring {pred_label_path}...")
    truth_path = UPath(truth_path)
    # Load the predicted and ground truth label volumes
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
        logging.info(f"Masking {label_name} with {mask_path}...")
        mask = zarr.open(mask_path.path, mode="r")[:]
        pred_label = pred_label * mask
        truth_label = truth_label * mask

    # Compute the scores
    if label_name in instance_classes:
        logging.info(
            f"Starting an instance evaluation for {label_name} in {crop_name}..."
        )
        timer = time()
        results = score_instance(pred_label, truth_label, crop.voxel_size)
        logging.info(
            f"Finished instance evaluation for {label_name} in {crop_name} in {time() - timer:.2f} seconds..."
        )
    else:
        results = score_semantic(pred_label, truth_label)
    results["num_voxels"] = int(np.prod(truth_label.shape))
    results["voxel_size"] = crop.voxel_size
    results["is_missing"] = False
    return crop_name, label_name, results


def empty_label_score(
    label, crop_name, instance_classes=INSTANCE_CLASSES, truth_path=TRUTH_PATH
):
    if label in instance_classes:
        truth_path = UPath(truth_path)
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
    submission_path = unzip_file(submission_path)

    # Find volumes to score
    logging.info(f"Scoring volumes in {submission_path}...")
    pred_volumes = [d.name for d in UPath(submission_path).glob("*") if d.is_dir()]
    truth_path = UPath(truth_path)
    logging.info(f"Volumes: {pred_volumes}")
    logging.info(f"Truth path: {truth_path}")
    truth_volumes = [d.name for d in truth_path.glob("*") if d.is_dir()]
    logging.info(f"Truth volumes: {truth_volumes}")

    found_volumes = list(set(pred_volumes) & set(truth_volumes))
    missing_volumes = list(set(truth_volumes) - set(pred_volumes))
    if len(found_volumes) == 0:
        raise ValueError(
            "No volumes found to score. Make sure the submission is formatted correctly."
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
    saved_path = UPath(zip_path).with_suffix(".zarr").path
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(saved_path)
        logging.info(f"Unzipped {zip_path} to {saved_path}")

    return UPath(saved_path)


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
