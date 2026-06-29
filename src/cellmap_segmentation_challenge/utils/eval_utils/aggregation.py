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

# Aggregate-level keys in a combined-scores dict; everything else is a per-crop entry.
AGGREGATE_KEYS = (
    "label_scores",
    "overall_instance_score",
    "overall_semantic_score",
    "overall_score",
    "total_evals",
    "num_evals_done",
    "git_version",
)


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

    # Pool raw counts per class: tp/fp/fn (F1 / IoU) and, for instance classes,
    # the per-instance Hausdorff sum/count (combined = sqrt(F1 * mean Hausdorff)).
    logging.info(f"Combining label scores...")
    scores = scores.copy()
    label_scores = {}
    counts = {}  # label -> {"tp", "fp", "fn"}
    haus = {}  # instance label -> {"sum": float, "count": int}
    for ds, these_scores in scores.items():
        # Skip aggregation-level keys that are not per-crop result dicts
        if not isinstance(these_scores, dict):
            continue
        if ds in AGGREGATE_KEYS:
            continue
        for label, this_score in these_scores.items():
            if this_score["is_missing"] and not include_missing:
                continue
            if label not in label_scores:
                label_scores[label] = {}
                counts[label] = {"tp": 0, "fp": 0, "fn": 0}
                if label in instance_classes:
                    haus[label] = {"sum": 0.0, "count": 0}
            for count_key in ("tp", "fp", "fn"):
                counts[label][count_key] += this_score[count_key]
            if label in instance_classes:
                haus[label]["sum"] += this_score["hausdorff_norm_sum"]
                haus[label]["count"] += this_score["n_hausdorff"]

    # Per-class scores; both instance and semantic overall are plain per-class means.
    instance_score_sum = 0.0
    instance_class_count = 0
    semantic_score_sum = 0.0
    semantic_class_count = 0
    for label in label_scores:
        tp, fp, fn = counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]
        if label in instance_classes:
            denom = 2 * tp + fp + fn
            count = haus[label]["count"]
            if denom == 0 and count == 0:  # nothing anywhere -> "nothing here"
                f1 = hausdorff = combined = 1.0
            else:
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                hausdorff = haus[label]["sum"] / count if count > 0 else 0.0
                combined = (f1 * hausdorff) ** 0.5
            label_scores[label]["f1"] = f1
            label_scores[label]["normalized_hausdorff_distance"] = hausdorff
            label_scores[label]["combined_score"] = combined
            instance_score_sum += combined  # plain per-class mean
            instance_class_count += 1
        else:
            denom = tp + fp + fn
            iou = (tp / denom) if denom > 0 else 1.0  # denom 0 = nothing anywhere
            label_scores[label]["iou"] = iou
            semantic_score_sum += iou  # plain per-class mean
            semantic_class_count += 1
        label_scores[label]["tp"] = tp
        label_scores[label]["fp"] = fp
        label_scores[label]["fn"] = fn
        for key in label_scores[label]:
            if label_scores[label][key] in cast_to_none:
                label_scores[label][key] = None
    scores["label_scores"] = label_scores

    logging.info("Computing overall scores...")
    scores["overall_instance_score"] = (
        instance_score_sum / instance_class_count if instance_class_count else 0
    )
    scores["overall_semantic_score"] = (
        semantic_score_sum / semantic_class_count if semantic_class_count else 0
    )
    scores["overall_score"] = (
        scores["overall_instance_score"] * scores["overall_semantic_score"]
    ) ** 0.5  # geometric mean

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


def public_scores(scores, submitted_labels):
    """Aggregate-only view for the participant-facing file.

    Keeps the overall scores and the per-class ratio scores (f1/iou/combined)
    for the submitted classes.

    Args:
        scores (dict): A combined-scores dict as returned by `combine_scores`.
        submitted_labels (set): Class names to include in the per-class scores.

    Returns:
        dict: Overall scores and per-class ratio scores.
    """
    public = {k: scores[k] for k in AGGREGATE_KEYS if k in scores}
    drop = ("tp", "fp", "fn")
    public["label_scores"] = {
        label: {k: v for k, v in s.items() if k not in drop}
        for label, s in scores.get("label_scores", {}).items()
        if label in submitted_labels
    }
    return public


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

        sanitize_scores(all_scores)
        # Classes with at least one real submission -> shown in the public file.
        submitted_labels = set(found_scores.get("label_scores", {}))

        # Public (participant-facing): aggregate-only, submitted classes.
        with open(result_file, "w") as f:
            json.dump(public_scores(all_scores, submitted_labels), f, indent=4)

        # Server-side (_extended): full scores with every per-crop count.
        extended_file = str(result_file).replace(
            UPath(result_file).suffix, "_extended" + UPath(result_file).suffix
        )
        with open(extended_file, "w") as f:
            json.dump(all_scores, f, indent=4)

        logging.info(
            f"Scores updated in {result_file} and {extended_file} in {time() - start_time:.2f} seconds"
        )
    else:
        logging.info("Final combined scores:")
        logging.info(all_scores)

    return all_scores, found_scores
