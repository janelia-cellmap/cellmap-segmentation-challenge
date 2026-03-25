"""Type definitions for evaluation metrics."""

from typing import Literal, TypedDict


class PQCropDict(TypedDict, total=False):
    """PQ accumulators returned per (crop, label) by score_label.

    Raw accumulators (tp/fp/fn/sum_iou) are summed globally across crops in
    ``combine_scores`` (micro-averaging) before per-category PQ/SQ/RQ values
    are derived.  The derived ``pq``, ``sq``, and ``f1`` fields are computed
    per-crop for interpretability and are NOT used in the global aggregation.
    """

    tp: int
    fp: int
    fn: int
    sum_iou: float
    pq: float  # per-crop PQ = sum_iou / (tp + 0.5*fp + 0.5*fn)
    sq: float  # per-crop SQ = sum_iou / tp (mean IoU of matched pairs; 0 if tp=0)
    f1: float  # per-crop F1/RQ = 2*tp / (2*tp + fp + fn)
    num_voxels: int
    voxel_size: tuple[float, ...]
    is_missing: bool
    status: Literal["scored", "skipped_too_many_instances", "missing"]
