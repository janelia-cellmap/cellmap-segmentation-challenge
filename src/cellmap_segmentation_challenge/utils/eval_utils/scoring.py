"""Core scoring functions for segmentation evaluation."""

import gc
import logging
from time import time

import cc3d
import numpy as np
import zarr
from fastremap import remap, unique
from upath import UPath

from ...config import TRUTH_PATH, INSTANCE_CLASSES
from ..crops import TEST_CROPS_DICT
from ..matched_crop import MatchedCrop
from .config import EvaluationConfig
from .distance import (
    compute_max_distance,
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


def _create_pathological_scores(status: str) -> InstanceScoreDict:
    """Create scores for a crop whose instance matching failed.

    The crop contributes nothing to the per-class pools (zero counts and no
    Hausdorff entries), so a failure neither helps nor penalizes the class.

    Args:
        status: Status string describing the failure.

    Returns:
        Score dict with zeroed counts and Hausdorff fields.
    """
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "hausdorff_norm_sum": 0.0,
        "n_hausdorff": 0,
        "status": status,
    }


def _compute_hausdorff_scores(
    mapping: dict[int, int],
    truth_label: np.ndarray,
    pred_label: np.ndarray,
    n_pred: int,
    voxel_size: tuple[float, ...],
    hausdorff_distance_max: float,
    truth_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-instance Hausdorff distances for pooling per class.

    Produces one distance per ground-truth instance (matched -> real distance,
    unmatched/FN -> max distance) plus a max-distance penalty per unmatched
    prediction (hallucination). Returns an empty array when the crop has no
    truth instances and no predictions, so empty crops contribute nothing to
    the per-class pool.

    Args:
        mapping: Instance ID mapping (pred -> truth)
        truth_label: Ground truth labels
        pred_label: Predicted labels (remapped to truth IDs)
        n_pred: Number of predicted instances
        voxel_size: Voxel size
        hausdorff_distance_max: Maximum distance
        truth_ids: Precomputed non-zero ground-truth ids

    Returns:
        1D array of per-instance Hausdorff distances (possibly empty)
    """
    # One distance per truth instance (matched -> real, unmatched/FN -> max).
    hausdorff_distances = optimized_hausdorff_distances(
        truth_label, pred_label, voxel_size, hausdorff_distance_max, truth_ids=truth_ids
    )

    # Max-distance penalty per unmatched prediction (hallucination).
    n_unmatched_pred = n_pred - len(set(mapping.keys()) - {0})
    if n_unmatched_pred > 0:
        hausdorff_distances = np.concatenate(
            [
                hausdorff_distances,
                np.full(n_unmatched_pred, hausdorff_distance_max, dtype=np.float32),
            ]
        )

    return hausdorff_distances


def score_instance(
    pred_label,
    truth_label,
    voxel_size,
    hausdorff_distance_max=None,
    config: EvaluationConfig | None = None,
) -> InstanceScoreDict:
    """Score instance segmentation against ground truth.

    Computes instance F1 score, Hausdorff distance, and combined metrics
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
        hausdorff_distance_max = compute_max_distance(voxel_size, truth_label.shape)
        logging.debug(
            f"Using default maximum Hausdorff distance of {hausdorff_distance_max:.2f}"
        )

    # Relabel predictions using connected components
    logging.info("Relabeling predicted instance labels...")
    # TODO: Switch to just renumbering to contiguous IDs, and leave user labels intact
    pred_label, n_pred = cc3d.connected_components(pred_label, return_N=True)

    # Match instances
    try:
        mapping = match_instances(truth_label, pred_label, config)
    except (TooManyInstancesError, TooManyOverlapEdgesError) as e:
        logging.warning(f"Instance matching failed: {e}")
        return _create_pathological_scores(
            hausdorff_distance_max,
            voxel_size,
            "skipped_too_many_instances",
        )
    except MatchingFailedError as e:
        logging.error(f"Matching optimization failed: {e}")
        return _create_pathological_scores(
            hausdorff_distance_max, voxel_size, "matching_failed"
        )

    # Remap predictions to match GT IDs
    if len(mapping) > 0 and not (len(mapping) == 1 and 0 in mapping):
        mapping[0] = 0  # background maps to background
        pred_label = remap(
            pred_label, mapping, in_place=True, preserve_missing_labels=True
        )

    # Non-zero ground-truth ids, computed once and shared with the Hausdorff step.
    truth_ids = unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]

    # Free matching-stage scratch before the memory-heavy Hausdorff phase.
    gc.collect()

    # Compute Hausdorff distances
    hausdorff_distances = _compute_hausdorff_scores(
        mapping, truth_label, pred_label, n_pred, voxel_size, hausdorff_distance_max, truth_ids
    )

    if len(hausdorff_distances) == 0:
        hausdorff_distances = [hausdorff_distance_max]

    # Compute F1 from instance matching counts
    gt_count = int(truth_ids.size)
    matched_gt_ids = set(mapping.values()) - {0}
    matched_pred_ids = set(mapping.keys()) - {0}

    tp = len(matched_gt_ids)
    fp = n_pred - len(matched_pred_ids)
    fn = gt_count - len(matched_gt_ids)
    if gt_count == 0 and n_pred == 0:
        # Correct true negative - 0/0, should be scored as 1.0
        f1 = 1.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)

    # Aggregate scores
    logging.info("Computing final scores...")
    hausdorff_distances = np.asarray(hausdorff_distances, dtype=np.float64)
    hausdorff_dist = float(hausdorff_distances.mean())
    # Normalize vectorized: norm(voxel_size) is constant; inf distance -> 0.
    vs_norm = np.linalg.norm(voxel_size)
    normalized_hausdorff_dist = float((1.01 ** (-hausdorff_distances / vs_norm)).mean())
    combined_score = (f1 * normalized_hausdorff_dist) ** 0.5
    logging.info(f"F1: {f1:.4f} (TP={tp}, FP={fp}, FN={fn})")
    logging.info(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    logging.info(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    logging.info(f"Combined Score: {combined_score:.4f}")

    return {
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hausdorff_distance": hausdorff_dist,
        "normalized_hausdorff_distance": normalized_hausdorff_dist,
        "combined_score": combined_score,
        "status": "scored",
    }


def score_semantic(pred_label, truth_label) -> dict[str, int | str]:
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

    # Voxel confusion counts; pooled per class in aggregation to compute IoU.
    tp = int(np.count_nonzero(truth_label & pred_label))
    fp = int(pred_label.sum()) - tp
    fn = int(truth_label.sum()) - tp

    logging.info(f"Semantic counts: TP={tp}, FP={fp}, FN={fn}")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "status": "scored",
    }


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
        gc.collect()

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
        # Penalize non-submission: every ground-truth instance is a false negative.
        truth_ids = unique(ds[:])
        n_instances = int(truth_ids[truth_ids != 0].size)
        return {
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": n_instances,
            "hausdorff_distance": compute_max_distance(voxel_size, ds.shape),
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
            "num_voxels": int(np.prod(ds.shape)),
            "voxel_size": voxel_size,
            "is_missing": True,
            "status": "missing",
        }
    else:
        # Not submitted: count every voxel as a false negative -> IoU 0.
        n_voxels = int(np.prod(ds.shape))
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_voxels,
            "num_voxels": n_voxels,
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
