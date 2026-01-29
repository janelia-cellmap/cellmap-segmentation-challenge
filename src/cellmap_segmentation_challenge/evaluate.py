import argparse
import gc
import json
import os
import shutil
from time import time
import zipfile

import numpy as np
import zarr
from scipy.spatial.distance import dice
from scipy.ndimage import distance_transform_edt
from fastremap import remap, unique, renumber
import cc3d
from cc3d.types import StatisticsDict, StatisticsSlicesDict

from zarr.errors import PathNotFoundError
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from upath import UPath

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .config import SUBMISSION_PATH, TRUTH_PATH, INSTANCE_CLASSES
from .utils import TEST_CROPS_DICT, MatchedCrop, rand_voi, get_git_hash

import logging
from typing import Optional, Protocol, Any, TypedDict, Literal
from dataclasses import dataclass, field


# ============================================================================
# Protocols for Dependency Injection
# ============================================================================


class ExecutorProtocol(Protocol):
    """Protocol for executor abstraction (ProcessPoolExecutor, ThreadPoolExecutor)."""

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Submit a callable to be executed."""
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        ...

    def __enter__(self) -> "ExecutorProtocol":
        """Context manager entry."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        ...


class ZarrStorageProtocol(Protocol):
    """Protocol for Zarr storage operations."""

    def open_group(self, path: str, mode: str = "r") -> zarr.Group:
        """Open a Zarr group."""
        ...

    def open_array(self, path: str, mode: str = "r") -> zarr.Array:
        """Open a Zarr array."""
        ...


class FileSystemProtocol(Protocol):
    """Protocol for filesystem operations."""

    def exists(self, path: UPath) -> bool:
        """Check if path exists."""
        ...

    def is_dir(self, path: UPath) -> bool:
        """Check if path is a directory."""
        ...

    def glob(self, path: UPath, pattern: str) -> list[UPath]:
        """Glob for files matching pattern."""
        ...


# ============================================================================
# Default Implementations
# ============================================================================


class FilesystemZarrStorage:
    """Default filesystem-based Zarr storage implementation."""

    def open_group(self, path: str, mode: str = "r") -> zarr.Group:
        """Open a Zarr group from filesystem."""
        return zarr.open(path, mode=mode)

    def open_array(self, path: str, mode: str = "r") -> zarr.Array:
        """Open a Zarr array from filesystem."""
        return zarr.open(path, mode=mode)


class StandardFileSystem:
    """Default filesystem implementation using UPath."""

    def exists(self, path: UPath) -> bool:
        """Check if path exists."""
        return path.exists()

    def is_dir(self, path: UPath) -> bool:
        """Check if path is a directory."""
        return path.is_dir()

    def glob(self, path: UPath, pattern: str) -> list[UPath]:
        """Glob for files matching pattern."""
        return list(path.glob(pattern))


# ============================================================================
# Type Definitions
# ============================================================================


class InstanceScoreDict(TypedDict, total=False):
    """Type definition for instance segmentation scores."""

    accuracy: float
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


# ============================================================================
# Custom Exceptions
# ============================================================================


class EvaluationError(Exception):
    """Base exception for evaluation errors."""

    pass


class TooManyInstancesError(EvaluationError):
    """Raised when submission has too many instances relative to ground truth.

    This is a pathological case where the ratio of predicted to ground truth
    instances exceeds acceptable thresholds, likely indicating poor segmentation.
    """

    def __init__(self, n_pred: int, n_gt: int, ratio: float, cutoff: float):
        self.n_pred = n_pred
        self.n_gt = n_gt
        self.ratio = ratio
        self.cutoff = cutoff
        super().__init__(
            f"Too many instances: {n_pred} predicted vs {n_gt} ground truth "
            f"(ratio: {ratio:.2f} exceeds cutoff: {cutoff:.2f})"
        )


class TooManyOverlapEdgesError(EvaluationError):
    """Raised when instance matching produces too many overlap edges.

    This indicates computational infeasibility for the matching algorithm.
    """

    def __init__(self, n_edges: int, max_edges: int):
        self.n_edges = n_edges
        self.max_edges = max_edges
        super().__init__(
            f"Too many overlap edges: {n_edges} exceeds maximum {max_edges}"
        )


class MatchingFailedError(EvaluationError):
    """Raised when instance matching optimization fails."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"Min-cost flow matching failed with status: {status}")


class ValidationError(EvaluationError):
    """Raised when input validation fails."""

    pass


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# Structured logging setup
import structlog
import psutil
from functools import wraps
from contextlib import contextmanager

# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get structured logger
struct_logger = structlog.get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline.

    All parameters can be set via environment variables or passed directly.
    Environment variables take precedence over defaults but not over
    explicitly passed values.
    """

    # Threading configuration
    max_instance_threads: int = 3
    max_semantic_threads: int = 25
    per_instance_threads: int = 25

    # Distance calculation parameters
    max_distance_cap_eps: float = 1e-4

    # Instance matching parameters
    final_instance_ratio_cutoff: float = 10.0
    initial_instance_ratio_cutoff: float = 50.0
    instance_ratio_factor: float = 5.0
    max_overlap_edges: int = 5_000_000
    mcmf_cost_scale: int = 1_000_000

    # Paths
    truth_path: UPath = field(default_factory=lambda: UPath(TRUTH_PATH))
    instance_classes: list[str] = field(default_factory=lambda: list(INSTANCE_CLASSES))

    # Values to cast to None in sanitization
    cast_to_none: list[Any] = field(
        default_factory=lambda: [np.nan, np.inf, -np.inf, float("inf"), float("-inf")]
    )

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Load configuration from environment variables with defaults.

        Returns:
            EvaluationConfig with values from environment or defaults.
        """
        return cls(
            max_instance_threads=int(os.getenv("MAX_INSTANCE_THREADS", "3")),
            max_semantic_threads=int(os.getenv("MAX_SEMANTIC_THREADS", "25")),
            per_instance_threads=int(os.getenv("PER_INSTANCE_THREADS", "25")),
            max_distance_cap_eps=float(os.getenv("MAX_DISTANCE_CAP_EPS", "1e-4")),
            final_instance_ratio_cutoff=float(
                os.getenv("FINAL_INSTANCE_RATIO_CUTOFF", "10")
            ),
            initial_instance_ratio_cutoff=float(
                os.getenv("INITIAL_INSTANCE_RATIO_CUTOFF", "50")
            ),
            instance_ratio_factor=float(os.getenv("INSTANCE_RATIO_FACTOR", "5.0")),
            max_overlap_edges=int(os.getenv("MAX_OVERLAP_EDGES", "5000000")),
            mcmf_cost_scale=int(os.getenv("MCMF_COST_SCALE", "1000000")),
        )

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.max_instance_threads < 1:
            raise ValueError(
                f"max_instance_threads must be >= 1, got {self.max_instance_threads}"
            )
        if self.max_semantic_threads < 1:
            raise ValueError(
                f"max_semantic_threads must be >= 1, got {self.max_semantic_threads}"
            )
        if self.per_instance_threads < 1:
            raise ValueError(
                f"per_instance_threads must be >= 1, got {self.per_instance_threads}"
            )
        if self.max_distance_cap_eps <= 0:
            raise ValueError(
                f"max_distance_cap_eps must be > 0, got {self.max_distance_cap_eps}"
            )
        if self.final_instance_ratio_cutoff <= 0:
            raise ValueError(
                f"final_instance_ratio_cutoff must be > 0, got {self.final_instance_ratio_cutoff}"
            )
        if self.initial_instance_ratio_cutoff <= 0:
            raise ValueError(
                f"initial_instance_ratio_cutoff must be > 0, got {self.initial_instance_ratio_cutoff}"
            )
        if self.instance_ratio_factor <= 0:
            raise ValueError(
                f"instance_ratio_factor must be > 0, got {self.instance_ratio_factor}"
            )
        if self.max_overlap_edges < 1:
            raise ValueError(
                f"max_overlap_edges must be >= 1, got {self.max_overlap_edges}"
            )
        if self.mcmf_cost_scale < 1:
            raise ValueError(
                f"mcmf_cost_scale must be >= 1, got {self.mcmf_cost_scale}"
            )


# ============================================================================
# Performance Monitoring Utilities
# ============================================================================


def timed_operation(operation_name: str):
    """Decorator to time and log function execution.

    Args:
        operation_name: Name of the operation for logging

    Example:
        >>> @timed_operation("instance_matching")
        >>> def match_instances(...):
        >>>     ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            try:
                result = func(*args, **kwargs)
                duration = time() - start
                struct_logger.info(
                    "operation_complete",
                    operation=operation_name,
                    function=func.__name__,
                    duration_seconds=duration,
                    status="success",
                )
                return result
            except Exception as e:
                duration = time() - start
                struct_logger.error(
                    "operation_failed",
                    operation=operation_name,
                    function=func.__name__,
                    duration_seconds=duration,
                    error=str(e),
                    error_type=type(e).__name__,
                    status="failed",
                )
                raise

        return wrapper

    return decorator


@contextmanager
def evaluation_context(operation: str, **context):
    """Context manager for tracking evaluation operations with resource monitoring.

    Args:
        operation: Name of the operation
        **context: Additional context to log

    Example:
        >>> with evaluation_context("scoring_instance", crop="crop1", label="mito"):
        >>>     scores = score_instance(...)
    """
    struct_logger.info(f"{operation}_started", **context)
    start_time = time()

    # Get initial resource usage
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    initial_cpu_percent = process.cpu_percent()

    try:
        yield
        duration = time() - start_time

        # Get final resource usage
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        final_cpu_percent = process.cpu_percent()

        struct_logger.info(
            f"{operation}_completed",
            duration_seconds=duration,
            memory_mb=final_memory_mb,
            memory_delta_mb=final_memory_mb - initial_memory_mb,
            cpu_percent=final_cpu_percent,
            **context,
        )
    except Exception as e:
        duration = time() - start_time
        final_memory_mb = process.memory_info().rss / 1024 / 1024

        struct_logger.error(
            f"{operation}_failed",
            duration_seconds=duration,
            memory_mb=final_memory_mb,
            error=str(e),
            error_type=type(e).__name__,
            **context,
        )
        raise


def log_resource_usage(operation: str, **context):
    """Log current resource usage.

    Args:
        operation: Name of the operation
        **context: Additional context to log
    """
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()

    struct_logger.info(
        "resource_usage",
        operation=operation,
        memory_mb=memory_mb,
        cpu_percent=cpu_percent,
        **context,
    )


@dataclass
class EvaluationMetrics:
    """Metrics for monitoring evaluation performance."""

    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_duration_seconds: float = 0.0
    avg_instance_score: float = 0.0
    avg_semantic_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "success_rate": self.successful_evaluations
            / max(self.total_evaluations, 1),
            "avg_duration_seconds": self.total_duration_seconds
            / max(self.total_evaluations, 1),
            "avg_instance_score": self.avg_instance_score,
            "avg_semantic_score": self.avg_semantic_score,
        }


# ============================================================================
# Legacy Constants (for backward compatibility during migration)
# ============================================================================

CAST_TO_NONE = [np.nan, np.inf, -np.inf, float("inf"), float("-inf")]
MAX_INSTANCE_THREADS = int(os.getenv("MAX_INSTANCE_THREADS", 3))
MAX_SEMANTIC_THREADS = int(os.getenv("MAX_SEMANTIC_THREADS", 25))
PER_INSTANCE_THREADS = int(os.getenv("PER_INSTANCE_THREADS", 25))
MAX_DISTANCE_CAP_EPS = float(os.getenv("MAX_DISTANCE_CAP_EPS", "1e-4"))
FINAL_INSTANCE_RATIO_CUTOFF = float(os.getenv("FINAL_INSTANCE_RATIO_CUTOFF", 10))
INITIAL_INSTANCE_RATIO_CUTOFF = float(os.getenv("INITIAL_INSTANCE_RATIO_CUTOFF", 50))
INSTANCE_RATIO_FACTOR = float(os.getenv("INSTANCE_RATIO_FACTOR", 5.0))
MAX_OVERLAP_EDGES = int(os.getenv("MAX_OVERLAP_EDGES", "5000000"))


def ratio_cutoff(
    nG: int,
    R_base: float = FINAL_INSTANCE_RATIO_CUTOFF,
    R_extra: float = INITIAL_INSTANCE_RATIO_CUTOFF,
    k: float = INSTANCE_RATIO_FACTOR,
) -> float:
    """Calculate the acceptable ratio cutoff for instance matching.

    The ratio cutoff decreases exponentially as the number of ground truth
    instances increases, allowing for more tolerance with fewer instances.

    Args:
        nG: Number of ground truth instances
        R_base: Base ratio cutoff (minimum)
        R_extra: Extra ratio tolerance for small nG
        k: Exponential decay factor

    Returns:
        Maximum acceptable ratio of predicted to ground truth instances
    """
    # nG==0 handled upstream (ratio undefined); return max tolerance for completeness
    return float(R_base + R_extra * np.exp(-nG / k))


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
    truth_stats = cc3d.statistics(truth_label)
    pred_stats = cc3d.statistics(pred_label)

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


def _compute_binary_metrics(
    truth_label: np.ndarray, pred_label: np.ndarray
) -> dict[str, float]:
    """Compute binary segmentation metrics.

    Args:
        truth_label: Ground truth labels
        pred_label: Predicted labels

    Returns:
        Dictionary with iou, dice_score, and binary_accuracy
    """
    truth_binary = (truth_label > 0).ravel()
    pred_binary = (pred_label > 0).ravel()

    iou = jaccard_score(truth_binary, pred_binary, zero_division=1)
    dice_score = 1 - dice(truth_binary, pred_binary)
    binary_accuracy = float((truth_binary == pred_binary).mean())

    return {
        "iou": iou,
        "dice_score": dice_score,
        "binary_accuracy": binary_accuracy,
    }


def _create_pathological_scores(
    binary_metrics: dict[str, float],
    voi_metrics: dict[str, float],
    hausdorff_distance_max: float,
    voxel_size: tuple[float, ...],
    status: str,
) -> InstanceScoreDict:
    """Create scores for pathological cases (matching failed).

    Args:
        binary_metrics: Pre-computed binary metrics
        voi_metrics: Pre-computed VoI metrics
        hausdorff_distance_max: Maximum Hausdorff distance
        voxel_size: Voxel size
        status: Status string for the failure

    Returns:
        Dictionary with worst-case scores
    """
    return {
        "accuracy": 0,
        "binary_accuracy": binary_metrics["binary_accuracy"],
        "hausdorff_distance": hausdorff_distance_max,
        "normalized_hausdorff_distance": normalize_distance(
            hausdorff_distance_max, voxel_size
        ),
        "combined_score": 0,
        "iou": binary_metrics["iou"],
        "dice_score": binary_metrics["dice_score"],
        "status": status,
        **voi_metrics,
    }


def _compute_hausdorff_scores(
    mapping: dict[int, int],
    truth_label: np.ndarray,
    pred_label: np.ndarray,
    n_pred: int,
    voxel_size: tuple[float, ...],
    hausdorff_distance_max: float,
) -> list[float]:
    """Compute Hausdorff distances for matched instances.

    Args:
        mapping: Instance ID mapping (pred -> truth)
        truth_label: Ground truth labels
        pred_label: Predicted labels (remapped to truth IDs)
        n_pred: Number of predicted instances
        voxel_size: Voxel size
        hausdorff_distance_max: Maximum distance

    Returns:
        List of Hausdorff distances
    """
    if len(mapping) == 1 and 0 in mapping:
        # Only background
        return [0.0]

    if len(mapping) > 0:
        # Compute Hausdorff for matched instances
        hausdorff_distances = optimized_hausdorff_distances(
            truth_label, pred_label, voxel_size, hausdorff_distance_max
        )

        # Add max distance for unmatched predictions
        matched_pred_ids = set(mapping.keys()) - {0}
        pred_ids = set(np.arange(1, n_pred + 1)) - {0}
        unmatched_pred = pred_ids - matched_pred_ids

        if len(unmatched_pred) > 0:
            hausdorff_distances = np.concatenate(
                [
                    hausdorff_distances,
                    np.full(
                        len(unmatched_pred), hausdorff_distance_max, dtype=np.float32
                    ),
                ]
            )

        return hausdorff_distances.tolist()
    else:
        # No matches
        return [hausdorff_distance_max]


def score_instance(
    pred_label,
    truth_label,
    voxel_size,
    hausdorff_distance_max=None,
    config: EvaluationConfig | None = None,
) -> InstanceScoreDict:
    """Score instance segmentation against ground truth.

    Computes pixel-wise accuracy, Hausdorff distance, and combined metrics
    after optimal instance matching.

    Args:
        pred_label: Predicted instance labels (0 = background)
        truth_label: Ground truth instance labels (0 = background)
        voxel_size: Physical voxel size in (Z, Y, X) order
        hausdorff_distance_max: Maximum Hausdorff distance cap (None = auto)
        config: Evaluation configuration (uses defaults if None)

    Returns:
        Dictionary containing all instance segmentation metrics

    Example:
        >>> scores = score_instance(pred, truth, voxel_size=(4.0, 4.0, 4.0))
        >>> print(f"Combined score: {scores['combined_score']:.3f}")
    """
    if config is None:
        config = EvaluationConfig.from_env()

    logging.info("Scoring instance segmentation...")

    # Determine Hausdorff distance cap
    if hausdorff_distance_max is None:
        hausdorff_distance_max = compute_default_max_distance(
            voxel_size, config.max_distance_cap_eps
        )
        logging.debug(
            f"Using default maximum Hausdorff distance of {hausdorff_distance_max:.2f}"
        )

    # Relabel predictions using connected components
    logging.info("Relabeling predicted instance labels...")
    pred_label, n_pred = cc3d.connected_components(pred_label, return_N=True)

    # Compute metrics that don't require matching
    binary_metrics = _compute_binary_metrics(truth_label, pred_label)
    voi = rand_voi(truth_label.astype(np.uint64), pred_label.astype(np.uint64))
    del voi["voi_split_i"], voi["voi_merge_j"]

    # Match instances
    try:
        mapping = match_instances(truth_label, pred_label, config)
    except (TooManyInstancesError, TooManyOverlapEdgesError) as e:
        logging.warning(f"Instance matching failed: {e}")
        return _create_pathological_scores(
            binary_metrics,
            voi,
            hausdorff_distance_max,
            voxel_size,
            "skipped_too_many_instances",
        )
    except MatchingFailedError as e:
        logging.error(f"Matching optimization failed: {e}")
        return _create_pathological_scores(
            binary_metrics, voi, hausdorff_distance_max, voxel_size, "matching_failed"
        )

    # Remap predictions to match GT IDs
    if len(mapping) > 0 and not (len(mapping) == 1 and 0 in mapping):
        mapping[0] = 0  # background maps to background
        pred_label = remap(
            pred_label, mapping, in_place=True, preserve_missing_labels=True
        )

    # Compute Hausdorff distances
    hausdorff_distances = _compute_hausdorff_scores(
        mapping, truth_label, pred_label, n_pred, voxel_size, hausdorff_distance_max
    )

    if len(hausdorff_distances) == 0:
        hausdorff_distances = [hausdorff_distance_max]

    # Aggregate scores
    logging.info("Computing final scores...")
    accuracy = float((truth_label == pred_label).mean())
    hausdorff_dist = float(np.mean(hausdorff_distances))
    normalized_hausdorff_dist = float(
        np.mean([normalize_distance(hd, voxel_size) for hd in hausdorff_distances])
    )
    combined_score = (accuracy * normalized_hausdorff_dist) ** 0.5

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Hausdorff Distance: {hausdorff_dist:.4f}")
    logging.info(f"Normalized Hausdorff Distance: {normalized_hausdorff_dist:.4f}")
    logging.info(f"Combined Score: {combined_score:.4f}")

    return {
        "accuracy": accuracy,
        "hausdorff_distance": hausdorff_dist,
        "normalized_hausdorff_distance": normalized_hausdorff_dist,
        "combined_score": combined_score,
        "status": "scored",
        **binary_metrics,
        **voi,
    }


def score_semantic(pred_label, truth_label) -> dict[str, float]:
    """
    Score a single semantic label volume against the ground truth semantic label volume.

    Args:
        pred_label (np.ndarray): The predicted semantic label volume.
        truth_label (np.ndarray): The ground truth semantic label volume.

    Returns:
        dict: A dictionary of scores for the semantic label volume.

    Example usage:
        scores = score_semantic(pred_label, truth_label)
    """
    logging.info("Scoring semantic segmentation...")
    # Flatten the label volumes and convert to binary
    pred_label = (pred_label > 0.0).ravel()
    truth_label = (truth_label > 0.0).ravel()

    # Compute the scores
    if np.sum(truth_label + pred_label) == 0:
        # If there are no true or false positives, set the scores to 1
        logging.debug("No true or false positives found. Setting scores to 1.")
        dice_score = 1
        iou_score = 1
    else:
        dice_score = 1 - dice(truth_label, pred_label)
        iou_score = jaccard_score(truth_label, pred_label, zero_division=1)
    scores = {
        "iou": iou_score,
        "dice_score": dice_score if not np.isnan(dice_score) else 1,
        "binary_accuracy": float((truth_label == pred_label).mean()),
        "status": "scored",
    }

    logging.info(f"IoU: {scores['iou']:.4f}")
    logging.info(f"Dice Score: {scores['dice_score']:.4f}")

    return scores


def score_label(
    pred_label_path,
    label_name,
    crop_name,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
):
    """
    Score a single label volume against the ground truth label volume.

    Args:
        pred_label_path (str): The path to the predicted label volume.
        truth_path (str): The path to the ground truth label volume.
        instance_classes (list): A list of instance classes.

    Returns:
        dict: A dictionary of scores for the label volume.

    Example usage:
        scores = score_label('pred.zarr/test_volume/label1')
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
    # drop big arrays before returning
    del truth_label, pred_label, truth_label_ds
    gc.collect()
    return crop_name, label_name, results


def empty_label_score(
    label, crop_name, instance_classes=INSTANCE_CLASSES, truth_path=TRUTH_PATH
):
    truth_path = UPath(truth_path)
    ds = zarr.open((truth_path / crop_name / label).path, mode="r")
    voxel_size = ds.attrs["voxel_size"]
    if label in instance_classes:
        truth_path = UPath(truth_path)
        return {
            "accuracy": 0,
            "hausdorff_distance": compute_default_max_distance(voxel_size),
            "normalized_hausdorff_distance": 0,
            "combined_score": 0,
            "num_voxels": int(np.prod(ds.shape)),
            "voxel_size": voxel_size,
            "is_missing": True,
            "status": "missing",
        }
    else:
        return {
            "iou": 0,
            "dice_score": 0,
            "num_voxels": int(np.prod(ds.shape)),
            "voxel_size": voxel_size,
            "is_missing": True,
            "status": "missing",
        }


def ensure_zgroup(path: UPath) -> zarr.Group:
    """
    Ensure that the given path can be opened as a zarr Group. If a .zgroup is not present, add it.
    """
    try:
        return zarr.open(path.path, mode="r")
    except PathNotFoundError:
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
        # Add a .zgroup file to force Zarr-2 format
        (path / ".zgroup").write_text('{"zarr_format": 2}')
        return zarr.open(path.path, mode="r")


def get_evaluation_args(
    volumes,
    submission_path,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
) -> list[tuple]:
    """
    Get the arguments for scoring each label in the submission.
    Args:
        volumes (list): A list of volumes to score.
        submission_path (str): The path to the submission volume.
        truth_path (str): The path to the ground truth volume.
        instance_classes (list): A list of instance classes.
    Returns:
        A list of tuples containing the arguments for each label to be scored.
    """
    if not isinstance(volumes, (tuple, list)):
        volumes = [volumes]
    score_label_arglist = []
    for volume in volumes:
        submission_path = UPath(submission_path)
        pred_volume_path = submission_path / volume
        logging.info(f"Scoring {pred_volume_path}...")
        truth_path = UPath(truth_path)

        # Find labels to score
        pred_labels = [a for a in ensure_zgroup(pred_volume_path).array_keys()]

        crop_name = pred_volume_path.name
        truth_labels = [a for a in ensure_zgroup(truth_path / crop_name).array_keys()]

        found_labels = list(set(pred_labels) & set(truth_labels))
        missing_labels = list(set(truth_labels) - set(pred_labels))

        # Score_label arguments for each label
        score_label_arglist.extend(
            [
                (
                    pred_volume_path / label if label in found_labels else None,
                    label,
                    crop_name,
                    truth_path,
                    instance_classes,
                )
                for label in truth_labels
            ]
        )
        logging.info(f"Missing labels: {missing_labels}")

    return score_label_arglist


def missing_volume_score(
    truth_path, volume, instance_classes=INSTANCE_CLASSES
) -> list[tuple]:
    """
    Score a missing volume as 0's, congruent with the score_volume function.

    Args:
        truth_path (str): The path to the ground truth volume.
        volume (str): The name of the volume.
        instance_classes (list): A list of instance classes.

    Returns:
        dict: A dictionary of scores for the volume.

    Example usage:
        scores = missing_volume_score('truth.zarr/test_volume')
    """
    logging.info(f"Scoring missing volume {volume}...")
    truth_path = UPath(truth_path)
    truth_volume_path = truth_path / volume

    # Find labels to score
    truth_labels = [a for a in ensure_zgroup(truth_volume_path).array_keys()]

    # Score each label
    scores = {
        label: empty_label_score(label, volume, instance_classes, truth_path)
        for label in truth_labels
    }

    return scores


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
                        "accuracy": 0,
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
            label_scores[label]["accuracy"] /= total_voxels[label]
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


def ensure_valid_submission(submission_path: UPath):
    """
    Ensure that the unzipped submission path is a valid Zarr-2 file.

    Args:
        submission_path (str): The path to the unzipped submission Zarr-2 file.

    Raises:
        ValueError: If the submission is not a valid unzipped Zarr-2 file.
    """
    # See if a Zarr was incorrectly zipped inside other folder(s)
    # If so, move contents from .zarr folder to submission_path and warn user
    zarr_folders = list(submission_path.glob("**/*.zarr"))
    if len(zarr_folders) == 0:
        # Try forcing Zarr-2 format by adding .zgroup if missing
        try:
            ensure_zgroup(submission_path)
            logging.warning(
                f"Submission at {submission_path} did not contain a .zgroup file. Added one to force Zarr-2 format."
            )
        except Exception as e:
            raise ValueError(
                f"Submission at {submission_path} is not a valid unzipped Zarr-2 file."
            ) from e
    elif len(zarr_folders) == 1:
        zarr_folder = zarr_folders[0]
        logging.warning(
            f"Submission at {submission_path} contains a Zarr folder inside subfolder(s) at {zarr_folder}. Moving contents to the root submission folder."
        )
        # Move contents of zarr_folder to submission_path
        for item in zarr_folder.iterdir():
            target = submission_path / item.name
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            shutil.move(str(item), str(submission_path))
        # Remove empty folders
        for parent in zarr_folder.parents:
            if parent == submission_path:
                break
            try:
                parent.rmdir()
            except OSError as e:
                logging.warning(
                    "Failed to remove directory %s while cleaning nested Zarr submission: %s",
                    parent,
                    e,
                )
        # Try opening again
        try:
            ensure_zgroup(submission_path)
            logging.warning(
                f"Submission at {submission_path} did not contain a .zgroup file. Added one to force Zarr-2 format."
            )
        except Exception as e:
            raise ValueError(
                f"Submission at {submission_path} is not a valid unzipped Zarr-2 file."
            ) from e
    elif len(zarr_folders) > 1:
        raise ValueError(
            f"Submission at {submission_path} contains multiple Zarr folders. Please ensure only one Zarr-2 file is submitted."
        )


def _prepare_submission(submission_path: UPath | str) -> UPath:
    """Unzip and validate submission.

    Args:
        submission_path: Path to zipped submission

    Returns:
        Path to unzipped, validated submission
    """
    unzipped_path = unzip_file(submission_path)
    ensure_valid_submission(UPath(unzipped_path))
    return UPath(unzipped_path)


def _discover_volumes(
    submission_path: UPath, truth_path: UPath
) -> tuple[list[str], list[str]]:
    """Discover volumes to score and missing volumes.

    Args:
                shutil.move(old_path, new_path)
        truth_path: Path to ground truth

    Returns:
        Tuple of (found_volumes, missing_volumes)

    Raises:
        ValueError: If no volumes found to score
    """
    pred_volumes = [d.name for d in submission_path.glob("*") if d.is_dir()]
    truth_volumes = [d.name for d in truth_path.glob("*") if d.is_dir()]

    found_volumes = list(set(pred_volumes) & set(truth_volumes))
    missing_volumes = list(set(truth_volumes) - set(pred_volumes))

    if len(found_volumes) == 0:
        # Check if "crop" prefixes are missing
        prefixed_pred_volumes = [f"crop{v}" for v in pred_volumes]
        found_volumes = list(set(prefixed_pred_volumes) & set(truth_volumes))

        if len(found_volumes) == 0:
            raise ValueError(
                "No volumes found to score. Make sure the submission is formatted correctly."
            )

        missing_volumes = list(set(truth_volumes) - set(prefixed_pred_volumes))

        # Rename predicted volumes to have "crop" prefix
        for v in pred_volumes:
            old_path = submission_path / v
            new_path = submission_path / f"crop{v}"
            try:
                old_path.rename(new_path)
            except Exception as exc:
                msg = (
                    f"Failed to rename predicted volume directory '{old_path}' to "
                    f"'{new_path}'. This may be due to missing files, insufficient "
                    "permissions, or an existing destination directory. Cannot "
                    "continue evaluation."
                )
                logging.error(msg)
                raise RuntimeError(msg) from exc

    return found_volumes, missing_volumes


def _execute_parallel_scoring(
    evaluation_args: list[tuple],
    config: EvaluationConfig,
) -> list[tuple]:
    """Execute evaluations in parallel using process pools.

    Args:
        evaluation_args: List of arguments for score_label
        config: Evaluation configuration

    Returns:
        List of (crop_name, label_name, result) tuples
    """
    instance_classes = config.instance_classes

    logging.info(
        f"Scoring volumes in parallel, using {config.max_instance_threads} "
        f"instance threads and {config.max_semantic_threads} semantic threads..."
    )

    # Use context managers for proper resource cleanup
    with (
        ProcessPoolExecutor(config.max_instance_threads) as instance_pool,
        ProcessPoolExecutor(config.max_semantic_threads) as semantic_pool,
    ):

        futures = []
        for args in evaluation_args:
            if args[1] in instance_classes:
                futures.append(instance_pool.submit(score_label, *args))
            else:
                futures.append(semantic_pool.submit(score_label, *args))

        results = []
        for future in tqdm(
            as_completed(futures),
            desc="Scoring volumes",
            total=len(futures),
            dynamic_ncols=True,
            leave=True,
        ):
            results.append(future.result())

    return results


def _aggregate_and_save_results(
    results: list[tuple],
    missing_scores: dict,
    result_file: str | None,
    config: EvaluationConfig,
) -> dict:
    """Aggregate results and optionally save to file.

    Args:
        results: List of (crop_name, label_name, result) tuples
        missing_scores: Scores for missing volumes
        result_file: Path to save results (None to skip saving)
        config: Evaluation configuration

    Returns:
        Dictionary of aggregated scores
    """
    scores = missing_scores.copy()

    # Process all results and update incrementally
    all_scores, found_scores = update_scores(
        scores, results, result_file, instance_classes=config.instance_classes
    )

    logging.info("Scores combined across all test volumes:")
    logging.info(
        f"\tOverall Instance Score: {all_scores['overall_instance_score']:.4f}"
    )
    logging.info(
        f"\tOverall Semantic Score: {all_scores['overall_semantic_score']:.4f}"
    )
    logging.info(f"\tOverall Score: {all_scores['overall_score']:.4f}")

    logging.info("Scores combined across test volumes with data submitted:")
    logging.info(
        f"\tOverall Instance Score: {found_scores['overall_instance_score']:.4f}"
    )
    logging.info(
        f"\tOverall Semantic Score: {found_scores['overall_semantic_score']:.4f}"
    )
    logging.info(f"\tOverall Score: {found_scores['overall_score']:.4f}")

    return all_scores


def score_submission(
    submission_path=UPath(SUBMISSION_PATH).with_suffix(".zip").path,
    result_file=None,
    truth_path=TRUTH_PATH,
    instance_classes=INSTANCE_CLASSES,
    config: EvaluationConfig | None = None,
):
    """Score a submission against the ground truth data.

    This is the main entry point for evaluating a submission. It unzips,
    validates, scores, and aggregates results for all volumes.

    Args:
        submission_path: Path to the zipped submission Zarr-2 file
        result_file: Path to save the scores (None to skip saving)
        truth_path: Path to the ground truth Zarr-2 file
        instance_classes: List of instance segmentation classes
        config: Evaluation configuration (uses defaults if None)

    Returns:
        Dictionary of aggregated scores across all volumes

    Raises:
        ValueError: If submission format is invalid
        RuntimeError: If volume renaming fails

    Example:
        >>> scores = score_submission('submission.zip', 'results.json')
        >>> print(f"Overall score: {scores['overall_score']:.4f}")

    Results structure:
        {
            "cropN": {  # Per-volume scores
                "label_name": {
                    # Instance segmentation
                    "accuracy": float,
                    "hausdorff_distance": float,
                    "combined_score": float,
                    # OR semantic segmentation
                    "iou": float,
                    "dice_score": float,
                }
            },
            "label_scores": {  # Aggregated per-label
                "label_name": {...}
            },
            "overall_instance_score": float,
            "overall_semantic_score": float,
            "overall_score": float,
        }
    """
    if config is None:
        config = EvaluationConfig.from_env()
        config.validate()

    # Override config with explicit parameters if provided
    if truth_path != TRUTH_PATH:
        config.truth_path = UPath(truth_path)
    if instance_classes != INSTANCE_CLASSES:
        config.instance_classes = list(instance_classes)

    logging.info(f"Scoring {submission_path}...")
    start_time = time()

    # Step 1: Prepare submission
    submission_path = _prepare_submission(submission_path)

    # Step 2: Discover volumes
    logging.info(f"Discovering volumes in {submission_path}...")
    found_volumes, missing_volumes = _discover_volumes(
        submission_path, config.truth_path
    )

    logging.info(f"Scoring volumes: {found_volumes}")
    if len(missing_volumes) > 0:
        logging.info(f"Missing volumes: {missing_volumes}")
        logging.info("Scoring missing volumes as 0's")

    # Step 3: Score missing volumes
    scores = {
        volume: missing_volume_score(
            config.truth_path, volume, instance_classes=config.instance_classes
        )
        for volume in missing_volumes
    }

    # Step 4: Get evaluation arguments
    evaluation_args = get_evaluation_args(
        found_volumes,
        submission_path=submission_path,
        truth_path=config.truth_path,
        instance_classes=config.instance_classes,
    )

    # Step 5: Execute parallel scoring
    results = _execute_parallel_scoring(evaluation_args, config)

    # Step 6: Aggregate and save results
    all_scores = _aggregate_and_save_results(results, scores, result_file, config)

    logging.info(f"Submission scored in {time() - start_time:.2f} seconds")

    if result_file is None:
        logging.info("Final combined scores:")
        logging.info(all_scores)

    return all_scores


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


def resize_array(arr, target_shape, pad_value=0):
    """
    Resize an array to a target shape by padding or cropping as needed.

    Parameters:
        arr (np.ndarray): Input array to resize.
        target_shape (tuple): Desired shape for the output array.
        pad_value (int, float, etc.): Value to use for padding if the array is smaller than the target shape.

    Returns:
        np.ndarray: Resized array with the specified target shape.
    """
    arr_shape = arr.shape
    resized_arr = arr

    # Pad if the array is smaller than the target shape
    pad_width = []
    for i in range(len(target_shape)):
        if arr_shape[i] < target_shape[i]:
            # Padding needed: calculate amount for both sides
            pad_before = (target_shape[i] - arr_shape[i]) // 2
            pad_after = target_shape[i] - arr_shape[i] - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            # No padding needed for this dimension
            pad_width.append((0, 0))

    if any(pad > 0 for pads in pad_width for pad in pads):
        resized_arr = np.pad(
            resized_arr, pad_width, mode="constant", constant_values=pad_value
        )

    # Crop if the array is larger than the target shape
    slices = []
    for i in range(len(target_shape)):
        if arr_shape[i] > target_shape[i]:
            # Calculate cropping slices to center the crop
            start = (arr_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            slices.append(slice(start, end))
        else:
            # No cropping needed for this dimension
            slices.append(slice(None))

    return resized_arr[tuple(slices)]


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


MAX_UNCOMPRESSED_SIZE = int(os.getenv("MAX_UNCOMPRESSED_SIZE", 50 * 1024**3))  # 50 GB


def _validate_zip_member(member: zipfile.ZipInfo, target_dir: str) -> None:
    """Validate a single zip member against path traversal and symlink attacks.

    Args:
        member: The zip entry to validate.
        target_dir: The resolved extraction directory.

    Raises:
        ValidationError: If the member is a symlink or would extract outside target_dir.
    """
    # Reject symlinks (external_attr upper 16 bits encode Unix mode; 0o120000 = symlink)
    if member.external_attr >> 16 & 0o170000 == 0o120000:
        raise ValidationError(
            f"Zip member {member.filename!r} is a symlink, which is not allowed."
        )

    # Resolve the destination and ensure it stays within target_dir and not equal to it
    target_real = os.path.realpath(target_dir)
    dest_real = os.path.realpath(os.path.join(target_real, member.filename))
    # Use commonpath to robustly ensure dest_real is a strict descendant of target_real
    if (
        os.path.commonpath([target_real, dest_real]) != target_real
        or dest_real == target_real
    ):
        raise ValidationError(
            f"Zip member {member.filename!r} would extract outside the target directory."
        )


def unzip_file(zip_path, max_uncompressed_size: int = MAX_UNCOMPRESSED_SIZE):
    """Unzip a zip file to a specified directory.

    Validates against path traversal (zip slip), symlink attacks, and
    decompression bombs before extracting.

    Args:
        zip_path (str): The path to the zip file.
        max_uncompressed_size (int): Maximum total uncompressed size in bytes.

    Raises:
        ValidationError: If any member fails security checks or total size exceeds limit.

    Example usage:
        unzip_file('submission.zip')
    """
    logging.info(f"Unzipping {zip_path}...")
    saved_path = UPath(zip_path).with_suffix(".zarr").path
    if UPath(saved_path).exists():
        logging.info(f"Using existing unzipped path at {saved_path}")
        return UPath(saved_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Check total uncompressed size (zip bomb guard)
        total_size = sum(info.file_size for info in zip_ref.infolist())
        if total_size > max_uncompressed_size:
            raise ValidationError(
                f"Zip uncompressed size ({total_size / 1024**3:.1f} GB) exceeds "
                f"limit ({max_uncompressed_size / 1024**3:.1f} GB)."
            )
        saved_path_real = os.path.realpath(saved_path)
        for member in zip_ref.infolist():
            _validate_zip_member(member, saved_path_real)
        for member in zip_ref.infolist():
            _validate_zip_member(member, saved_path)

        zip_ref.extractall(saved_path)

    logging.info(f"Unzipped {zip_path} to {saved_path}")

    return UPath(saved_path)


if __name__ == "__main__":
    # When called on the commandline, evaluate the submission
    # example usage: python evaluate.py submission.zip
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
