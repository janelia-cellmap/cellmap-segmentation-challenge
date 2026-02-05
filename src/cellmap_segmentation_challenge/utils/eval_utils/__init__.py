"""Evaluation utilities for cellmap segmentation challenge.

This module provides all evaluation-related functionality including:
- Scoring for instance and semantic segmentation
- Instance matching using min-cost flow optimization
- Hausdorff distance and other metrics
- Submission processing and validation
"""

# Types
from .types import InstanceScoreDict, SemanticScoreDict

# Exceptions
from .exceptions import (
    EvaluationError,
    TooManyInstancesError,
    TooManyOverlapEdgesError,
    MatchingFailedError,
    ValidationError,
)

# Configuration
from .config import (
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
)

# Instance matching
from .instance_matching import (
    InstanceOverlapData,
    match_instances,
)

# Distance metrics
from .distance import (
    compute_default_max_distance,
    normalize_distance,
    optimized_hausdorff_distances,
    bbox_for_label,
    roi_slices_for_pair,
    compute_hausdorff_distance_roi,
)

# Scoring functions
from .scoring import (
    score_instance,
    score_semantic,
    score_label,
    empty_label_score,
    match_crop_space,
)

# Score aggregation
from .aggregation import (
    combine_scores,
    sanitize_scores,
    update_scores,
    num_evals_done,
)

# Submission processing
from .submission import (
    ensure_zgroup,
    ensure_valid_submission,
    get_evaluation_args,
    missing_volume_score,
    score_submission,
)

# Array utilities
from .array_utils import resize_array

# Zip utilities
from .zip_utils import (
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
    "compute_default_max_distance",
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
