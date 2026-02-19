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
    total_voxels = {}
    for ds, these_scores in scores.items():
        for label, this_score in these_scores.items():
            # logging.info(this_score)
            if this_score["is_missing"] and not include_missing:
                continue
            if label in instance_classes:
                if label not in label_scores:
                    label_scores[label] = {
                        "mean_accuracy": 0,
                        "hausdorff_distance": 0,
                        "normalized_hausdorff_distance": 0,
                        "combined_score": 0,
                    }
                    total_voxels[label] = 0
            else:
                if label not in label_scores:
                    label_scores[label] = {"iou": 0, "dice_score": 0}
                    total_voxels[label] = 0
            for key in label_scores[label].keys():
                if this_score[key] is None:
                    continue
                label_scores[label][key] += this_score[key] * this_score["num_voxels"]
                if this_score[key] in cast_to_none:
                    scores[ds][label][key] = None
            total_voxels[label] += this_score["num_voxels"]

    # Normalize back to the total number of voxels
    for label in label_scores:
        if label in instance_classes:
            label_scores[label]["mean_accuracy"] /= total_voxels[label]
            label_scores[label]["hausdorff_distance"] /= total_voxels[label]
            label_scores[label]["normalized_hausdorff_distance"] /= total_voxels[label]
            label_scores[label]["combined_score"] /= total_voxels[label]
        else:
            label_scores[label]["iou"] /= total_voxels[label]
            label_scores[label]["dice_score"] /= total_voxels[label]
        # Cast to None if the value is in `cast_to_none`
        for key in label_scores[label]:
            if label_scores[label][key] in cast_to_none:
                label_scores[label][key] = None
    scores["label_scores"] = label_scores

    # Compute the overall score
    logging.info("Computing overall scores...")
    overall_instance_scores = []
    overall_semantic_scores = []
    instance_total_voxels = sum(
        total_voxels[label] for label in label_scores if label in instance_classes
    )
    semantic_total_voxels = sum(
        total_voxels[label] for label in label_scores if label not in instance_classes
    )
    for label in label_scores:
        if label in instance_classes:
            overall_instance_scores += [
                label_scores[label]["combined_score"] * total_voxels[label]
            ]
        else:
            overall_semantic_scores += [
                label_scores[label]["iou"] * total_voxels[label]
            ]
    scores["overall_instance_score"] = (
        np.nansum(overall_instance_scores) / instance_total_voxels
        if overall_instance_scores
        else 0
    )
    scores["overall_semantic_score"] = (
        np.nansum(overall_semantic_scores) / semantic_total_voxels
        if overall_semantic_scores
        else 0
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
