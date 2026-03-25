"""Configuration for evaluation pipeline."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from upath import UPath

from ...config import SUBMISSION_PATH, TRUTH_PATH, INSTANCE_CLASSES

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline.

    All parameters can be set via environment variables or passed directly.
    Environment variables take precedence over defaults but not over
    explicitly passed values.

    With the PQ metric, both instance and semantic scoring are dominated by
    zarr I/O rather than CPU, so a single unified worker pool replaces the
    old separate instance / semantic pools.  ``MAX_WORKERS`` (env var) sets
    the pool size; the legacy ``MAX_INSTANCE_THREADS`` and
    ``MAX_SEMANTIC_THREADS`` variables are still honoured as a fallback so
    that existing deployment configs keep working.
    """

    # Unified worker pool (replaces separate instance + semantic pools)
    max_workers: int = 32

    # Distance calculation parameters (kept for distance.py, not used in scoring)
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

        ``MAX_WORKERS`` takes precedence.  If absent, falls back to
        ``MAX_INSTANCE_THREADS + MAX_SEMANTIC_THREADS`` for backward
        compatibility with existing deployment configs.

        Returns:
            EvaluationConfig with values from environment or defaults.
        """
        # Resolve worker count: prefer MAX_WORKERS, fall back to legacy sum
        legacy_sum = int(os.getenv("MAX_INSTANCE_THREADS", "3")) + int(
            os.getenv("MAX_SEMANTIC_THREADS", "25")
        )
        max_workers = int(os.getenv("MAX_WORKERS", str(legacy_sum)))

        return cls(
            max_workers=max_workers,
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
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
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


# Legacy constants — kept for backward compatibility
CAST_TO_NONE = [np.nan, np.inf, -np.inf, float("inf"), float("-inf")]
MAX_INSTANCE_THREADS = int(os.getenv("MAX_INSTANCE_THREADS", 3))
MAX_SEMANTIC_THREADS = int(os.getenv("MAX_SEMANTIC_THREADS", 25))
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
