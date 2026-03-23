"""Score aggregation and result processing utilities."""

import json
import logging
from time import time

import numpy as np
from upath import UPath

from ...config import INSTANCE_CLASSES
from ..crops import TEST_CROPS_DICT
from ..utils import get_git_hash
from .config import CAST_TO_NONE


def combine_scores(
    scores,
    include_missing=True,
    instance_classes=INSTANCE_CLASSES,
    cast_to_none=CAST_TO_NONE,
):
    """Combine PQ accumulators across all crops and compute per-category PQ/SQ/RQ.

    PQ is computed per-crop-label and then **macro-averaged** (unweighted mean)
    across crops for each category.  This ensures that missing volumes and
    FP-heavy submissions are treated symmetrically: both contribute PQ=0 for
    that crop regardless of the FP count, so omitting a volume cannot inflate
    the score relative to submitting (wrong) predictions.

    SQ and RQ are still computed from globally-accumulated TP/FP/FN (micro-
    averaged) for informational purposes only and are not used in ranking.

    Per-category metrics:

    * PQ_c = mean over crops of [sum_IoU / (TP + 0.5·FP + 0.5·FN)]
    * SQ_c = global_sum_IoU / global_TP   (Segmentation Quality; 0 when TP=0)
    * RQ_c = global_TP / (global_TP + 0.5·global_FP + 0.5·global_FN)

    The final scores are **unweighted** means of PQ_c across categories:

    * ``overall_thing_pq``  — mean over thing (instance) categories
    * ``overall_stuff_pq``  — mean over stuff (semantic) categories
    * ``overall_score``     — mean over all categories

    ``overall_instance_score`` and ``overall_semantic_score`` are kept as
    legacy aliases for downstream consumers.

    Args:
        scores: Dict mapping crop names to per-label score dicts.
        include_missing: Whether to include missing-submission crops.
        instance_classes: List of thing-class label names.
        cast_to_none: Unused; kept for call-site compatibility.

    Returns:
        The input ``scores`` dict augmented with ``label_scores``,
        ``overall_thing_pq``, ``overall_stuff_pq``, ``overall_score``,
        and the legacy alias keys.
    """
    logging.info("Combining label scores (PQ)...")
    scores = scores.copy()

    # Per-crop PQ lists for macro-averaging; global accum for SQ/RQ/informational
    pq_lists: dict[str, list[float]] = {}
    accum: dict[str, dict] = {}
    for crop_name, crop_scores in scores.items():
        if not isinstance(crop_scores, dict):
            continue
        for label, score in crop_scores.items():
            if not isinstance(score, dict) or "tp" not in score:
                continue
            if score.get("is_missing") and not include_missing:
                continue
            tp, fp, fn, sum_iou = (
                score["tp"],
                score["fp"],
                score["fn"],
                score["sum_iou"],
            )
            # Per-crop PQ (0 when denom=0, e.g. missing volume or empty GT)
            denom = tp + 0.5 * fp + 0.5 * fn
            pq_crop = sum_iou / denom if denom > 0 else 0.0
            if label not in pq_lists:
                pq_lists[label] = []
            pq_lists[label].append(pq_crop)
            # Global accumulation for SQ/RQ
            if label not in accum:
                accum[label] = {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
            accum[label]["tp"] += tp
            accum[label]["fp"] += fp
            accum[label]["fn"] += fn
            accum[label]["sum_iou"] += sum_iou

    # Compute PQ (macro) / SQ / RQ per category
    logging.info("Computing per-category PQ/SQ/RQ...")
    label_scores: dict[str, dict] = {}
    for label in pq_lists:
        # PQ: macro-averaged across crops
        pq = float(np.mean(pq_lists[label])) if pq_lists[label] else 0.0
        # SQ / RQ: micro-averaged (global) — informational only
        acc = accum.get(label, {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0})
        tp, fp, fn, sum_iou = acc["tp"], acc["fp"], acc["fn"], acc["sum_iou"]
        denom = tp + 0.5 * fp + 0.5 * fn
        sq = sum_iou / tp if tp > 0 else 0.0
        rq = tp / denom if denom > 0 else 0.0
        label_scores[label] = {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sum_iou": sum_iou,
        }
    scores["label_scores"] = label_scores

    # Unweighted mean PQ broken down by thing / stuff
    logging.info("Computing overall scores...")
    thing_pqs = [label_scores[l]["pq"] for l in label_scores if l in instance_classes]
    stuff_pqs = [
        label_scores[l]["pq"] for l in label_scores if l not in instance_classes
    ]
    all_pqs = thing_pqs + stuff_pqs

    scores["overall_thing_pq"] = float(np.mean(thing_pqs)) if thing_pqs else 0.0
    scores["overall_stuff_pq"] = float(np.mean(stuff_pqs)) if stuff_pqs else 0.0
    scores["overall_score"] = float(np.mean(all_pqs)) if all_pqs else 0.0

    # Legacy aliases expected by update_scores and downstream consumers
    scores["overall_instance_score"] = scores["overall_thing_pq"]
    scores["overall_semantic_score"] = scores["overall_stuff_pq"]

    return scores


def num_evals_done(all_scores):
    num_evals_done = 0
    for volume, scores in all_scores.items():
        if "crop" in volume:
            num_evals_done += len(scores.keys())
    return num_evals_done


def sanitize_scores(scores):
    """
    Sanitize scores by converting NaN values to None.

    Args:
        scores (dict): A dictionary of scores.

    Returns:
        dict: A sanitized dictionary of scores.
    """
    for volume, volume_scores in scores.items():
        if isinstance(volume_scores, dict):
            for label, label_scores in volume_scores.items():
                if isinstance(label_scores, dict):
                    for key, value in label_scores.items():
                        if value is None:
                            continue
                        if isinstance(value, str):
                            continue
                        if not np.isscalar(value) and len(value) == 1:
                            value = value[0]
                        if np.isscalar(value):
                            if np.isnan(value) or np.isinf(value) or np.isneginf(value):
                                scores[volume][label][key] = None
                            elif isinstance(value, np.floating):
                                scores[volume][label][key] = float(value)
                        else:
                            if any(
                                [
                                    np.isnan(v) or np.isinf(v) or np.isneginf(v)
                                    for v in value
                                ]
                            ):
                                scores[volume][label][key] = None
                            elif any([isinstance(v, np.floating) for v in value]):
                                scores[volume][label][key] = [float(v) for v in value]
    return scores


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
    all_scores["git_version"] = get_git_hash()

    found_scores = combine_scores(
        scores, include_missing=False, instance_classes=instance_classes
    )

    if result_file is not None:
        logging.info(f"Saving collected scores to {result_file}...")

        with open(result_file, "w") as f:
            json.dump(sanitize_scores(all_scores), f, indent=4)

        found_result_file = str(result_file).replace(
            UPath(result_file).suffix, "_submitted_only" + UPath(result_file).suffix
        )
        with open(found_result_file, "w") as f:
            json.dump(sanitize_scores(found_scores), f, indent=4)

        logging.info(
            f"Scores updated in {result_file} and {found_result_file} in {time() - start_time:.2f} seconds"
        )
    else:
        logging.info("Final combined scores:")
        logging.info(all_scores)

    return all_scores, found_scores
