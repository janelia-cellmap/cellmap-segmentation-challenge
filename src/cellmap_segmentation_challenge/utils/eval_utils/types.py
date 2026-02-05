"""Type definitions for evaluation metrics."""

from typing import Literal, TypedDict


class InstanceScoreDict(TypedDict, total=False):
    """Type definition for instance segmentation scores."""

    mean_accuracy: float
    binary_accuracy: float
    hausdorff_distance: float
    normalized_hausdorff_distance: float
    combined_score: float
    iou: float
    dice_score: float
    num_voxels: int
    voxel_size: tuple[float, ...]
    is_missing: bool
    status: Literal["scored", "skipped_too_many_instances", "missing"]
    voi_split: float
    voi_merge: float


class SemanticScoreDict(TypedDict, total=False):
    """Type definition for semantic segmentation scores."""

    iou: float
    dice_score: float
    binary_accuracy: float
    num_voxels: int
    voxel_size: tuple[float, ...]
    is_missing: bool
    status: Literal["scored", "missing"]
