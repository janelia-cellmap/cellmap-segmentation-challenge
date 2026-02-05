"""Distance metrics including Hausdorff distance computation."""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from cc3d import statistics as cc3d_statistics
from cc3d.types import StatisticsDict, StatisticsSlicesDict
from fastremap import unique
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .config import EvaluationConfig, PER_INSTANCE_THREADS


def compute_default_max_distance(
    voxel_size,
    eps: float | None = None,
    config: Optional["EvaluationConfig"] = None,
) -> float:
    """
    Compute the default maximum distance used for distance-based metrics.

    If ``eps`` is not provided, the value is taken from ``config.max_distance_cap_eps``.
    If both ``eps`` and ``config`` are ``None``, a default ``EvaluationConfig`` is
    created via ``EvaluationConfig.from_env()``.
    """
    if eps is None:
        if config is None:
            config = EvaluationConfig.from_env()
        eps = config.max_distance_cap_eps

    v = np.linalg.norm(np.asarray(voxel_size, dtype=float))
    return float(v * (np.log(1.0 / eps) / np.log(1.01)))


def normalize_distance(distance: float, voxel_size) -> float:
    """
    Normalize a distance value to [0, 1] using the maximum distance represented by a voxel
    """
    if distance == np.inf:
        return 0.0
    return float((1.01 ** (-distance / np.linalg.norm(voxel_size))))


def optimized_hausdorff_distances(
    truth_label,
    pred_label,
    voxel_size,
    hausdorff_distance_max,
    method="standard",
    percentile: float | None = None,
):
    """
    Compute per-truth-instance Hausdorff-like distances against the (already remapped)
    prediction using multithreading. Returns a 1D float32 numpy array whose i-th
    entry corresponds to truth_ids[i].

    Parameters
    ----------
    truth_label : np.ndarray
        Ground-truth instance label volume (0 == background).
    pred_label : np.ndarray
        Prediction instance label volume that has already been remapped to align
        with the GT ids (0 == background).
    voxel_size : Sequence[float]
        Physical voxel sizes in Z, Y, X (or Y, X) order.
    hausdorff_distance_max : float
        Cap for distances (use np.inf for uncapped).
    method : {"standard", "modified", "percentile"}
        "standard" -> classic Hausdorff (max of directed maxima)
        "modified" -> mean of directed distances, then max of the two means
        "percentile" -> use the given percentile of directed distances (requires
                         `percentile` to be provided).
    percentile : float | None
        Percentile (0-100) used when method=="percentile".
    """
    # Unique GT ids (exclude background = 0)
    truth_ids = unique(truth_label)
    truth_ids = truth_ids[truth_ids != 0]
    true_num = int(truth_ids.size)
    if true_num == 0:
        return np.empty((0,), dtype=np.float32)

    voxel_size = np.asarray(voxel_size, dtype=np.float64)
    truth_stats = cc3d_statistics(truth_label)
    pred_stats = cc3d_statistics(pred_label)

    def get_distance(i: int):
        tid = int(truth_ids[i])
        h_dist = compute_hausdorff_distance_roi(
            truth_label,
            truth_stats,
            pred_label,
            pred_stats,
            tid,
            voxel_size,
            hausdorff_distance_max,
            method=method,
            percentile=percentile,
        )
        return i, float(h_dist)

    dists = np.empty((true_num,), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=PER_INSTANCE_THREADS) as executor:
        for idx, h in tqdm(
            executor.map(get_distance, range(true_num)),
            desc="Computing Hausdorff distances",
            total=true_num,
            dynamic_ncols=True,
        ):
            dists[idx] = h

    return dists


def bbox_for_label(
    stats: StatisticsDict | StatisticsSlicesDict,
    ndim: int,
    label_id: int,
):
    """
    Try to get bbox without allocating a full boolean mask using cc3d statistics.
    Falls back to mask-based bbox if cc3d doesn't provide expected fields.
    Returns (mins, maxs) inclusive-exclusive in voxel indices, or None if missing.
    """
    # stats = cc3d.statistics(label_vol)
    # cc3d.statistics usually returns dict-like with keys per label id.
    # There are multiple API variants; try common patterns.
    if "bounding_boxes" in stats:
        # bounding_boxes is a list where index corresponds to label_id
        bounding_boxes = stats["bounding_boxes"]
        if label_id >= len(bounding_boxes):
            return None
        bb = bounding_boxes[label_id]
        if bb is None:
            return None
        # bb is a tuple of slices, convert to (mins, maxs)
        if isinstance(bb, tuple) and all(isinstance(s, slice) for s in bb):
            mins = [s.start for s in bb]
            maxs = [s.stop for s in bb]
            return mins, maxs
        # bb might be (z0,z1,y0,y1,x0,x1) with end exclusive
        mins = [bb[2 * k] for k in range(ndim)]
        maxs = [bb[2 * k + 1] for k in range(ndim)]
        return mins, maxs

    if label_id in stats:
        s = stats[label_id]
        if "bounding_box" in s:
            bb = s["bounding_box"]
            mins = [bb[2 * k] for k in range(ndim)]
            maxs = [bb[2 * k + 1] for k in range(ndim)]
            return mins, maxs


def roi_slices_for_pair(
    truth_stats: StatisticsDict | StatisticsSlicesDict,
    pred_stats: StatisticsDict | StatisticsSlicesDict,
    tid: int,
    voxel_size,
    ndim: int,
    shape: tuple[int, ...],
    max_distance: float,
):
    """
    ROI = union(bbox(truth==tid), bbox(pred==tid)) padded by P derived from max_distance.
    Returns tuple of slices suitable for numpy indexing.
    """
    vs = np.asarray(voxel_size, dtype=float)
    if vs.size != ndim:
        # tolerate vs longer (e.g. includes channel), take last ndim
        vs = vs[-ndim:]

    # padding per axis in voxels
    pad = np.ceil(max_distance / vs).astype(int) + 2

    tb = bbox_for_label(truth_stats, ndim, tid)
    assert tb is not None, f"Truth ID {tid} not found in truth statistics."

    tmins, tmaxs = tb
    pb = bbox_for_label(pred_stats, ndim, tid)
    if pb is None:
        pmins, pmaxs = tmins, tmaxs
    else:
        pmins, pmaxs = pb

    mins = [min(tmins[d], pmins[d]) for d in range(ndim)]
    maxs = [max(tmaxs[d], pmaxs[d]) for d in range(ndim)]

    # expand and clamp
    out_slices = []
    for d in range(ndim):
        a = max(0, mins[d] - int(pad[d]))
        b = min(shape[d], maxs[d] + int(pad[d]))
        out_slices.append(slice(a, b))
    return tuple(out_slices)


def compute_hausdorff_distance_roi(
    truth_label: np.ndarray,
    truth_stats: StatisticsDict | StatisticsSlicesDict,
    pred_label: np.ndarray,
    pred_stats: StatisticsDict | StatisticsSlicesDict,
    tid: int,
    voxel_size,
    max_distance: float,
    method: str = "standard",
    percentile: float | None = None,
):
    """
    Same metric as compute_hausdorff_distance(), but operates on an ROI slice
    and builds masks only inside ROI.
    """
    ndim = truth_label.ndim

    roi = roi_slices_for_pair(
        truth_stats,
        pred_stats,
        tid,
        voxel_size,
        ndim,
        truth_label.shape,
        max_distance,
    )

    t_roi = truth_label[roi]
    p_roi = pred_label[roi]

    a = t_roi == tid
    b = p_roi == tid

    a_n = int(a.sum())
    b_n = int(b.sum())
    if a_n == 0 and b_n == 0:
        return 0.0
    elif a_n == 0 or b_n == 0:
        return max_distance

    vs = np.asarray(voxel_size, dtype=np.float64)
    if vs.size != ndim:
        vs = vs[-ndim:]

    dist_to_b = distance_transform_edt(~b, sampling=vs)
    dist_to_a = distance_transform_edt(~a, sampling=vs)

    fwd = dist_to_b[a]
    bwd = dist_to_a[b]

    if method == "standard":
        d = max(fwd.max(initial=0.0), bwd.max(initial=0.0))
    elif method == "modified":
        d = max(fwd.mean() if fwd.size else 0.0, bwd.mean() if bwd.size else 0.0)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("'percentile' must be provided when method='percentile'")
        d = max(
            float(np.percentile(fwd, percentile)) if fwd.size else 0.0,
            float(np.percentile(bwd, percentile)) if bwd.size else 0.0,
        )
    else:
        raise ValueError("method must be one of {'standard', 'modified', 'percentile'}")

    return float(min(d, max_distance))
