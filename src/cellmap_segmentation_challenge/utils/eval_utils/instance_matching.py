"""Instance matching using min-cost flow optimization."""

import logging
from dataclasses import dataclass

import numpy as np

from .config import EvaluationConfig, ratio_cutoff
from .exceptions import (
    TooManyInstancesError,
    TooManyOverlapEdgesError,
    MatchingFailedError,
)


@dataclass
class InstanceOverlapData:
    """Data structure for instance overlap computation."""

    nG: int  # Number of ground truth instances
    nP: int  # Number of predicted instances
    rows: np.ndarray  # GT indices for overlaps
    cols: np.ndarray  # Pred indices for overlaps
    iou_vals: np.ndarray  # IoU values for overlaps


def _check_instance_counts(nG: int, nP: int) -> bool:
    """Check if instance counts allow matching.

    Args:
        nG: Number of ground truth instances
        nP: Number of predicted instances

    Returns:
        True if matching should proceed, False if special case handled
    """
    if (nG == 0 and nP > 0) or (nP == 0 and nG > 0):
        if nG == 0 and nP > 0:
            logging.info("No GT instances; returning empty match.")
        if nP == 0 and nG > 0:
            logging.info("No Pred instances; returning empty match.")
        return False
    elif nG == 0 and nP == 0:
        logging.info("No GT or Pred instances; returning only background match.")
        return False
    return True


def _check_instance_ratio(nP: int, nG: int, config: EvaluationConfig) -> None:
    """Check if predicted/GT ratio is within acceptable bounds.

    Args:
        nP: Number of predicted instances
        nG: Number of ground truth instances
        config: Evaluation configuration

    Raises:
        TooManyInstancesError: If ratio exceeds cutoff
    """
    assert nG > 0, "nG must be > 0 to check instance ratio"
    cutoff = ratio_cutoff(
        nG,
        config.final_instance_ratio_cutoff,
        config.initial_instance_ratio_cutoff,
        config.instance_ratio_factor,
    )
    ratio = nP / nG

    if ratio > cutoff:
        logging.warning(
            f"Instance ratio {ratio:.2f} exceeds cutoff {cutoff:.2f} "
            f"({nP} pred vs {nG} GT)"
        )
        raise TooManyInstancesError(nP, nG, ratio, cutoff)


def _compute_instance_overlaps(
    gt: np.ndarray, pred: np.ndarray, nG: int, nP: int, max_edges: int
) -> InstanceOverlapData:
    """Compute IoU overlaps between GT and predicted instances.

    Args:
        gt: Ground truth instance labels (1D or flattened view)
        pred: Predicted instance labels (1D or flattened view)
        nG: Number of ground truth instances
        nP: Number of predicted instances
        max_edges: Maximum number of overlap edges allowed

    Returns:
        InstanceOverlapData with overlap information

    Raises:
        TooManyOverlapEdgesError: If number of edges exceeds max_edges
    """
    # 1D views
    g = np.ravel(gt)
    p = np.ravel(pred)

    # Foreground masks
    g_fg = g > 0
    p_fg = p > 0
    fg = g_fg & p_fg

    # Per-object sizes
    gt_sizes = np.bincount((g[g_fg].astype(np.int64) - 1), minlength=nG)[:, None]
    pr_sizes = np.bincount((p[p_fg].astype(np.int64) - 1), minlength=nP)[None, :]

    # Foreground overlaps
    gi = g[fg].astype(np.int64) - 1
    pj = p[fg].astype(np.int64) - 1

    if gi.size == 0:
        # No overlaps
        return InstanceOverlapData(
            nG=nG,
            nP=nP,
            rows=np.array([], dtype=np.int64),
            cols=np.array([], dtype=np.int64),
            iou_vals=np.array([], dtype=np.float32),
        )

    # Encode pairs and count
    gi_u = gi.astype(np.uint64)
    pj_u = pj.astype(np.uint64)
    key = gi_u * np.uint64(nP) + pj_u

    uniq_keys, counts = np.unique(key, return_counts=True)

    if uniq_keys.size > max_edges:
        raise TooManyOverlapEdgesError(uniq_keys.size, max_edges)

    rows = (uniq_keys // np.uint64(nP)).astype(np.int64)
    cols = (uniq_keys % np.uint64(nP)).astype(np.int64)

    # Compute IoU
    inter = counts.astype(np.int64)
    union = gt_sizes[rows, 0] + pr_sizes[0, cols] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        iou_vals = (inter / union).astype(np.float32)

    # Keep only IoU > 0
    keep = iou_vals > 0.0
    rows = rows[keep]
    cols = cols[keep]
    iou_vals = iou_vals[keep]

    return InstanceOverlapData(nG=nG, nP=nP, rows=rows, cols=cols, iou_vals=iou_vals)


def _solve_matching_problem(
    overlap_data: InstanceOverlapData, cost_scale: int
) -> dict[int, int]:
    """Solve min-cost flow matching problem.

    Args:
        overlap_data: Instance overlap data
        cost_scale: Scale factor for cost values

    Returns:
        Dictionary mapping predicted ID to ground truth ID

    Raises:
        MatchingFailedError: If optimization fails
    """
    from ortools.graph.python import min_cost_flow

    nG = overlap_data.nG
    nP = overlap_data.nP
    rows = overlap_data.rows
    cols = overlap_data.cols
    iou_vals = overlap_data.iou_vals

    mcf = min_cost_flow.SimpleMinCostFlow()

    # Node indexing
    source = 0
    gt0 = 1
    pred0 = gt0 + nG
    sink = pred0 + nP

    UNMATCH_COST = cost_scale + 1

    # Build arcs
    tails = []
    heads = []
    caps = []
    costs = []

    def add_arc(u: int, v: int, cap: int, cost: int) -> None:
        tails.append(u)
        heads.append(v)
        caps.append(cap)
        costs.append(cost)

    # Source -> GT
    for i in range(nG):
        add_arc(source, gt0 + i, 1, 0)

    # GT -> Sink (unmatched option)
    for i in range(nG):
        add_arc(gt0 + i, sink, 1, UNMATCH_COST)

    # GT -> Pred edges
    for r, c, iou in zip(rows, cols, iou_vals):
        u = gt0 + int(r)
        v = pred0 + int(c)
        cost = int((1.0 - float(iou)) * cost_scale)
        add_arc(u, v, 1, cost)

    # Pred -> Sink
    for j in range(nP):
        add_arc(pred0 + j, sink, 1, 0)

    # Add arcs in bulk
    mcf.add_arcs_with_capacity_and_unit_cost(
        np.asarray(tails, dtype=np.int32),
        np.asarray(heads, dtype=np.int32),
        np.asarray(caps, dtype=np.int64),
        np.asarray(costs, dtype=np.int64),
    )

    # Set supplies
    mcf.set_node_supply(source, int(nG))
    mcf.set_node_supply(sink, -int(nG))

    # Solve
    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise MatchingFailedError(status)

    # Extract matches
    mapping: dict[int, int] = {}
    for a in range(mcf.num_arcs()):
        if mcf.flow(a) != 1:
            continue
        u = mcf.tail(a)
        v = mcf.head(a)
        if gt0 <= u < pred0 and pred0 <= v < sink:
            gt_id = (u - gt0) + 1
            pred_id = (v - pred0) + 1
            mapping[pred_id] = gt_id

    return mapping


def match_instances(
    gt: np.ndarray,
    pred: np.ndarray,
    config: EvaluationConfig | None = None,
) -> dict[int, int]:
    """Match instances between GT and prediction based on IoU.

    Uses min-cost flow optimization to find optimal 1:1 matching between
    predicted and ground truth instances based on IoU overlap.

    Args:
        gt: Ground truth instance labels (0 = background)
        pred: Predicted instance labels (0 = background)
        config: Evaluation configuration (uses defaults if None)

    Returns:
        Dictionary mapping predicted instance ID to ground truth instance ID.
        Returns {0: 0} if only background present.
        Returns {} if no matches found or one side has no instances.

    Raises:
        ValidationError: If array shapes don't match
        TooManyInstancesError: If pred/GT ratio exceeds threshold
        TooManyOverlapEdgesError: If overlap computation is too large
        MatchingFailedError: If optimization fails

    Example:
        >>> mapping = match_instances(gt, pred)
        >>> # Remap predictions to match GT IDs
        >>> pred_aligned = remap(pred, mapping, preserve_missing_labels=True)
    """
    if config is None:
        config = EvaluationConfig.from_env()

    # Get instance counts
    g = np.ravel(gt)
    p = np.ravel(pred)
    nG = int(g.max()) if g.size else 0
    nP = int(p.max()) if p.size else 0

    # Check for special cases
    if not _check_instance_counts(nG, nP):
        if nG == 0 and nP == 0:
            return {0: 0}
        return {}

    # Check instance ratio
    _check_instance_ratio(nP, nG, config)

    # Compute overlaps
    overlap_data = _compute_instance_overlaps(
        gt, pred, nG, nP, config.max_overlap_edges
    )

    # Handle case of no overlaps
    if overlap_data.rows.size == 0:
        return {}

    # Solve matching problem
    mapping = _solve_matching_problem(overlap_data, config.mcmf_cost_scale)

    return mapping
