"""Evaluation module for cellmap segmentation challenge.

This module provides evaluation functionality for instance and semantic
segmentation tasks. All implementation details are organized in
utils/eval_utils/ for better maintainability.

Example usage:
    >>> from cellmap_segmentation_challenge.evaluate import score_submission
    >>> scores = score_submission('submission.zip', 'results.json')
    >>> print(f"Overall score: {scores['overall_score']:.4f}")
"""

# Re-export all public APIs for backward compatibility
from .utils.eval_utils import (
    # Types
    InstanceScoreDict,
    SemanticScoreDict,
    # Exceptions
    EvaluationError,
    TooManyInstancesError,
    TooManyOverlapEdgesError,
    MatchingFailedError,
    ValidationError,
    # Configuration
    EvaluationConfig,
    CAST_TO_NONE,
    MAX_INSTANCE_THREADS,
    MAX_SEMANTIC_THREADS,
    PER_INSTANCE_THREADS,
    MAX_DISTANCE_CAP_EPS,
    FINAL_INSTANCE_RATIO_CUTOFF,
    INITIAL_INSTANCE_RATIO_CUTOFF,
    INSTANCE_RATIO_FACTOR,
    MAX_OVERLAP_EDGES,
    ratio_cutoff,
    # Instance matching
    InstanceOverlapData,
    match_instances,
    # Distance metrics
    compute_max_distance,
    normalize_distance,
    optimized_hausdorff_distances,
    bbox_for_label,
    roi_slices_for_pair,
    compute_hausdorff_distance_roi,
    # Scoring functions
    score_instance,
    score_semantic,
    score_label,
    empty_label_score,
    match_crop_space,
    # Score aggregation
    combine_scores,
    sanitize_scores,
    update_scores,
    num_evals_done,
    # Submission processing
    ensure_zgroup,
    ensure_valid_submission,
    get_evaluation_args,
    missing_volume_score,
    score_submission,
    # Array utilities
    resize_array,
    # Zip utilities
    MAX_UNCOMPRESSED_SIZE,
    unzip_file,
)

__all__ = [
    # Types
    "InstanceScoreDict",
    "SemanticScoreDict",
    # Exceptions
    "EvaluationError",
    "TooManyInstancesError",
    "TooManyOverlapEdgesError",
    "MatchingFailedError",
    "ValidationError",
    # Configuration
    "EvaluationConfig",
    "CAST_TO_NONE",
    "MAX_INSTANCE_THREADS",
    "MAX_SEMANTIC_THREADS",
    "PER_INSTANCE_THREADS",
    "MAX_DISTANCE_CAP_EPS",
    "FINAL_INSTANCE_RATIO_CUTOFF",
    "INITIAL_INSTANCE_RATIO_CUTOFF",
    "INSTANCE_RATIO_FACTOR",
    "MAX_OVERLAP_EDGES",
    "ratio_cutoff",
    # Instance matching
    "InstanceOverlapData",
    "match_instances",
    # Distance metrics
    "compute_max_distance",
    "normalize_distance",
    "optimized_hausdorff_distances",
    "bbox_for_label",
    "roi_slices_for_pair",
    "compute_hausdorff_distance_roi",
    # Scoring functions
    "score_instance",
    "score_semantic",
    "score_label",
    "empty_label_score",
    "match_crop_space",
    # Score aggregation
    "combine_scores",
    "sanitize_scores",
    "update_scores",
    "num_evals_done",
    # Submission processing
    "ensure_zgroup",
    "ensure_valid_submission",
    "get_evaluation_args",
    "missing_volume_score",
    "score_submission",
    # Array utilities
    "resize_array",
    # Zip utilities
    "MAX_UNCOMPRESSED_SIZE",
    "unzip_file",
]


if __name__ == "__main__":
    # CLI entry point
    import argparse
    from .config import TRUTH_PATH

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
