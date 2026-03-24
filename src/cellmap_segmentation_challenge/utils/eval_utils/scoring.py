"""Core scoring functions for segmentation evaluation."""

import gc
import logging
from time import time

import cc3d
import numpy as np
import zarr
from upath import UPath

from ...config import TRUTH_PATH, INSTANCE_CLASSES
from ..crops import TEST_CROPS_DICT
from ..matched_crop import MatchedCrop
from .config import EvaluationConfig
from .exceptions import TooManyInstancesError, TooManyOverlapEdgesError
from .instance_matching import match_instances_pq


def score_instance(
    pred_label,
    truth_label,
    voxel_size,
    hausdorff_distance_max=None,
    config: EvaluationConfig | None = None,
) -> dict:
    """Score instance segmentation against ground truth using Panoptic Quality.

    Relabels predictions via connected components, then matches GT and predicted
    instances with IoU > 0.5 (greedy, descending-IoU order).  Returns raw PQ
    accumulators for this crop; they are summed globally in ``combine_scores``.

    Args:
        pred_label: Predicted instance labels (0 = background).
        truth_label: Ground truth instance labels (0 = background).
        voxel_size: Physical voxel size — kept for call-site compatibility,
            not used internally.
        hausdorff_distance_max: Unused; kept for call-site compatibility.
        config: Evaluation configuration (uses defaults if None).

    Returns:
        Dict with keys ``tp``, ``fp``, ``fn``, ``sum_iou``, ``status``.
    """
    if config is None:
        config = EvaluationConfig.from_env()

    logging.info("Scoring instance segmentation (PQ)...")

    # Relabel predictions using connected components
    logging.info("Relabeling predicted instance labels...")
    pred_label, _ = cc3d.connected_components(pred_label, return_N=True)

    nG = int(truth_label.max()) if truth_label.size else 0
    nP = int(pred_label.max()) if pred_label.size else 0

    try:
        tp, fp, fn, sum_iou = match_instances_pq(
            truth_label,
            pred_label,
            max_edges=config.max_overlap_edges,
        )
    except (TooManyInstancesError, TooManyOverlapEdgesError) as e:
        logging.warning(f"PQ matching skipped: {e}")
        return {
            "tp": 0,
            "fp": nP,
            "fn": nG,
            "sum_iou": 0.0,
            "status": "skipped_too_many_instances",
        }

    logging.info(f"TP={tp}, FP={fp}, FN={fn}, sum_IoU={sum_iou:.4f}")
    return {"tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou, "status": "scored"}


def score_semantic(pred_label, truth_label) -> dict:
    """Score a semantic (stuff) label volume using Panoptic Quality.

    Collapses GT and prediction each to a single binary segment per crop
    (at most one GT segment and one predicted segment).  Computes their IoU;
    if IoU > 0.5 it counts as a TP match, otherwise as FP + FN.

    Args:
        pred_label: Predicted semantic label volume.
        truth_label: Ground truth semantic label volume.

    Returns:
        Dict with keys ``tp``, ``fp``, ``fn``, ``sum_iou``, ``status``.
    """
    logging.info("Scoring semantic segmentation (PQ)...")

    gt_present = bool(np.any(truth_label > 0))
    pred_present = bool(np.any(pred_label > 0))

    if not gt_present and not pred_present:
        return {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0, "status": "scored"}
    if not gt_present:
        return {"tp": 0, "fp": 1, "fn": 0, "sum_iou": 0.0, "status": "scored"}
    if not pred_present:
        return {"tp": 0, "fp": 0, "fn": 1, "sum_iou": 0.0, "status": "scored"}

    # Both present — compute binary IoU of the single collapsed segments
    gt_bin = truth_label > 0
    pred_bin = pred_label > 0
    intersection = int(np.count_nonzero(gt_bin & pred_bin))
    union = int(np.count_nonzero(gt_bin | pred_bin))
    iou = intersection / union  # union > 0 guaranteed (both non-empty)

    logging.info(f"Semantic IoU={iou:.4f}")

    if iou > 0.5:
        return {"tp": 1, "fp": 0, "fn": 0, "sum_iou": float(iou), "status": "scored"}
    else:
        return {"tp": 0, "fp": 1, "fn": 1, "sum_iou": 0.0, "status": "scored"}


def _pq_f1_from_accum(tp, fp, fn, sum_iou):
    """Compute per-crop PQ, SQ, and F1 from raw accumulators.

    F1 (= RQ, Recognition Quality) measures detection accuracy:
        F1 = 2·TP / (2·TP + FP + FN)

    SQ (Segmentation Quality) is the mean IoU of matched instances:
        SQ = sum_IoU / TP  (0.0 when TP=0)

    PQ (Panoptic Quality) = SQ × RQ, equivalently:
        PQ = sum_IoU / (TP + 0.5·FP + 0.5·FN)

    All are 0.0 when their denominator is zero.
    """
    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_denom if f1_denom > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    pq_denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / pq_denom if pq_denom > 0 else 0.0
    return float(pq), float(sq), float(f1)


def score_label(
    pred_label_path,
    label_name,
    crop_name,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
):
    """Score a single label volume against the ground truth label volume.

    Args:
        pred_label_path: Path to the predicted label volume (or None if missing).
        label_name: Name of the label (e.g. ``"mito"``).
        crop_name: Name of the crop (e.g. ``"crop42"``).
        truth_path: Path to the ground truth zarr store.
        instance_classes: List of instance (thing) class names.

    Returns:
        Tuple ``(crop_name, label_name, results_dict)``.
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
    pq, sq, f1 = _pq_f1_from_accum(
        results["tp"], results["fp"], results["fn"], results["sum_iou"]
    )
    results["pq"] = pq
    results["sq"] = sq
    results["f1"] = f1
    # drop big arrays before returning
    del truth_label, pred_label, truth_label_ds
    gc.collect()
    return crop_name, label_name, results


def empty_label_score(
    label, crop_name, instance_classes=INSTANCE_CLASSES, truth_path=TRUTH_PATH
):
    """Return worst-case PQ accumulators for a label missing from the submission.

    For thing (instance) classes the number of FN equals the number of GT
    instances.  For stuff (semantic) classes FN is 1 if the GT contains any
    foreground, else 0.

    Args:
        label: Label name.
        crop_name: Crop name.
        instance_classes: List of instance (thing) class names.
        truth_path: Path to the ground truth zarr store.

    Returns:
        PQ accumulator dict with ``is_missing=True``.
    """
    truth_path = UPath(truth_path)
    ds = zarr.open((truth_path / crop_name / label).path, mode="r")
    voxel_size = ds.attrs["voxel_size"]
    num_voxels = int(np.prod(ds.shape))
    arr = ds[:]

    if label in instance_classes:
        fn = int(arr.max()) if arr.size else 0  # number of GT instances
    else:
        fn = 1 if np.any(arr > 0) else 0  # one stuff segment, or nothing

    pq, sq, f1 = _pq_f1_from_accum(0, 0, fn, 0.0)
    return {
        "tp": 0,
        "fp": 0,
        "fn": fn,
        "sum_iou": 0.0,
        "pq": pq,
        "sq": sq,
        "f1": f1,
        "num_voxels": num_voxels,
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
