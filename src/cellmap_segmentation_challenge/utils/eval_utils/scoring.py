"""Core scoring functions for segmentation evaluation."""

import gc
import logging
from time import time

import cc3d
import numpy as np
import zarr
from fastremap import remap
from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_score
from upath import UPath

from ...config import TRUTH_PATH, INSTANCE_CLASSES
from ..crops import TEST_CROPS_DICT
from ..matched_crop import MatchedCrop
from ..rand_voi import rand_voi
from .config import EvaluationConfig
from .distance import (
    compute_default_max_distance,
    normalize_distance,
    optimized_hausdorff_distances,
)
from .exceptions import (
    TooManyInstancesError,
    TooManyOverlapEdgesError,
    MatchingFailedError,
)
from .instance_matching import match_instances
from .types import InstanceScoreDict


def _compute_binary_metrics(
    truth_label: np.ndarray, pred_label: np.ndarray
) -> dict[str, float]:
    """Compute binary segmentation metrics.

    Args:
        truth_label: Ground truth labels
        pred_label: Predicted labels

    Returns:
        Dictionary with iou, dice_score, and binary_accuracy
    """
    truth_binary = (truth_label > 0).ravel()
    pred_binary = (pred_label > 0).ravel()

    iou = jaccard_score(truth_binary, pred_binary, zero_division=1)
    dice_score = 1 - dice(truth_binary, pred_binary)
    binary_accuracy = float((truth_binary == pred_binary).mean())

    return {
        "iou": iou,
        "dice_score": dice_score,
        "binary_accuracy": binary_accuracy,
    }


def _create_pathological_scores(
    binary_metrics: dict[str, float],
    voi_metrics: dict[str, float],
    hausdorff_distance_max: float,
    voxel_size: tuple[float, ...],
    status: str,
) -> InstanceScoreDict:
    """Create scores for pathological cases (matching failed).

    Args:
        binary_metrics: Pre-computed binary metrics
        voi_metrics: Pre-computed VoI metrics
        hausdorff_distance_max: Maximum Hausdorff distance
        voxel_size: Voxel size
        status: Status string for the failure

    Returns:
        Dictionary with worst-case scores
    """
    return {
        "mean_accuracy": 0,
        "binary_accuracy": binary_metrics["binary_accuracy"],
        "hausdorff_distance": hausdorff_distance_max,
        "normalized_hausdorff_distance": normalize_distance(
            hausdorff_distance_max, voxel_size
        ),
        "combined_score": 0,
        "iou": binary_metrics["iou"],
        "dice_score": binary_metrics["dice_score"],
        "status": status,
        **voi_metrics,
    }


def _compute_hausdorff_scores(
    mapping: dict[int, int],
    truth_label: np.ndarray,
    pred_label: np.ndarray,
    n_pred: int,
    voxel_size: tuple[float, ...],
    hausdorff_distance_max: float,
) -> list[float]:
    """Compute Hausdorff distances for matched instances.

    Args:
        mapping: Instance ID mapping (pred -> truth)
        truth_label: Ground truth labels
        pred_label: Predicted labels (remapped to truth IDs)
        n_pred: Number of predicted instances
        voxel_size: Voxel size
        hausdorff_distance_max: Maximum distance

    Returns:
        List of Hausdorff distances
    """
    if len(mapping) == 1 and 0 in mapping:
        # Only background
        return [0.0]

    if len(mapping) > 0:
        # Compute Hausdorff for matched instances
        hausdorff_distances = optimized_hausdorff_distances(
            truth_label, pred_label, voxel_size, hausdorff_distance_max
        )

        # Add max distance for unmatched predictions
        matched_pred_ids = set(mapping.keys()) - {0}
        pred_ids = set(np.arange(1, n_pred + 1)) - {0}
        unmatched_pred = pred_ids - matched_pred_ids

        if len(unmatched_pred) > 0:
            hausdorff_distances = np.concatenate(
                [
                    hausdorff_distances,
                    np.full(
                        len(unmatched_pred), hausdorff_distance_max, dtype=np.float32
                    ),
                ]
            )

        return hausdorff_distances.tolist()
    else:
        # No matches
        return [hausdorff_distance_max]


def score_instance(
    pred_label,
    truth_label,
    voxel_size,
    hausdorff_distance_max=None,
    config: EvaluationConfig | None = None,
) -> InstanceScoreDict:
    """Score instance segmentation against ground truth.

    Computes pixel-wise accuracy, Hausdorff distance, and combined metrics
    after optimal instance matching.

    Args:
        pred_label: Predicted instance labels (0 = background)
        truth_label: Ground truth instance labels (0 = background)
        voxel_size: Physical voxel size in (Z, Y, X) order
        hausdorff_distance_max: Maximum Hausdorff distance cap (None = auto)
        config: Evaluation configuration (uses defaults if None)

    Returns:
        Dictionary containing all instance segmentation metrics

    Example:
        >>> scores = score_instance(pred, truth, voxel_size=(4.0, 4.0, 4.0))
        >>> print(f"Combined score: {scores['combined_score']:.3f}")
    """
    if config is None:
        config = EvaluationConfig.from_env()

    logging.info("Scoring instance segmentation...")

    # Determine Hausdorff distance cap
    if hausdorff_distance_max is None:
        hausdorff_distance_max = compute_default_max_distance(
            voxel_size, config.max_distance_cap_eps
        )
        logging.debug(
            f"Using default maximum Hausdorff distance of {hausdorff_distance_max:.2f}"
        )

    # Relabel predictions using connected components
    logging.info("Relabeling predicted instance labels...")
    pred_label, n_pred = cc3d.connected_components(pred_label, return_N=True)

    # Compute metrics that don't require matching
    binary_metrics = _compute_binary_metrics(truth_label, pred_label)
    voi = rand_voi(truth_label.astype(np.uint64), pred_label.astype(np.uint64))
    del voi["voi_split_i"], voi["voi_merge_j"]

    # Match instances
    try:
        mapping = match_instances(truth_label, pred_label, config)
    except (TooManyInstancesError, TooManyOverlapEdgesError) as e:
        logging.warning(f"Instance matching failed: {e}")
        return _create_pathological_scores(
            binary_metrics,
            voi,
            hausdorff_distance_max,
            voxel_size,
            "skipped_too_many_instances",
        )
    except MatchingFailedError as e:
        logging.error(f"Matching optimization failed: {e}")
        return _create_pathological_scores(
            binary_metrics, voi, hausdorff_distance_max, voxel_size, "matching_failed"
        )

    # Remap predictions to match GT IDs
    if len(mapping) > 0 and not (len(mapping) == 1 and 0 in mapping):
        mapping[0] = 0  # background maps to background
        pred_label = remap(
            pred_label, mapping, in_place=True, preserve_missing_labels=True
        )

    # Compute Hausdorff distances
    hausdorff_distances = _compute_hausdorff_scores(
        mapping, truth_label, pred_label, n_pred, voxel_size, hausdorff_distance_max
    )

    if len(hausdorff_distances) == 0:
        hausdorff_distances = [hausdorff_distance_max]

    # Aggregate scores
    logging.info("Computing final scores...")
    mean_accuracy = float((truth_label == pred_label).mean())
    hausdorff_dist = float(np.mean(hausdorff_distances))
    normalized_hausdorff_dist = float(
        np.mean([normalize_distance(hd, voxel_size) for hd in hausdorff_distances])
    )
    combined_score = (mean_accuracy * normalized_hausdorff_dist) ** 0.5
    logging.info(f"Mean Accuracy: {mean_accuracy:.4f}")
    logging.info(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    logging.info(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    logging.info(f"Combined Score: {combined_score:.4f}")

    return {
        "mean_accuracy": mean_accuracy,
        "hausdorff_distance": hausdorff_dist,
        "normalized_hausdorff_distance": normalized_hausdorff_dist,
        "combined_score": combined_score,
        "status": "scored",
        **binary_metrics,
        **voi,
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
    pred_label = (pred_label > 0.0).ravel()
    truth_label = (truth_label > 0.0).ravel()

    # Compute the scores
    if np.sum(truth_label + pred_label) == 0:
        # If there are no true or false positives, set the scores to 1
        logging.debug("No true or false positives found. Setting scores to 1.")
        dice_score = 1
        iou_score = 1
    else:
        dice_score = 1 - dice(truth_label, pred_label)
        iou_score = jaccard_score(truth_label, pred_label, zero_division=1)
    scores = {
        "iou": iou_score,
        "dice_score": dice_score if not np.isnan(dice_score) else 1,
        "binary_accuracy": float((truth_label == pred_label).mean()),
        "status": "scored",
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
):
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
        del mask

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
    # drop big arrays before returning
    del truth_label, pred_label, truth_label_ds
    gc.collect()
    return crop_name, label_name, results


def empty_label_score(
    label, crop_name, instance_classes=INSTANCE_CLASSES, truth_path=TRUTH_PATH
):
    truth_path = UPath(truth_path)
    ds = zarr.open((truth_path / crop_name / label).path, mode="r")
    voxel_size = ds.attrs["voxel_size"]
    if label in instance_classes:
        truth_path = UPath(truth_path)
        return {
            "mean_accuracy": 0,
            "hausdorff_distance": compute_default_max_distance(voxel_size),
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
            "num_voxels": int(np.prod(ds.shape)),
            "voxel_size": voxel_size,
            "is_missing": True,
            "status": "missing",
        }
    else:
        return {
            "iou": 0,
            "dice_score": 0,
            "binary_accuracy": 0,
            "num_voxels": int(np.prod(ds.shape)),
            "voxel_size": voxel_size,
            "is_missing": True,
            "status": "missing",
        }


def match_crop_space(path, class_label, voxel_size, shape, translation) -> np.ndarray:
    mc = MatchedCrop(
        path=path,
        class_label=class_label,
        target_voxel_size=voxel_size,
        target_shape=shape,
        target_translation=translation,
        instance_classes=INSTANCE_CLASSES,
        semantic_threshold=0.5,
        pad_value=0,
    )
    return mc.load_aligned()
