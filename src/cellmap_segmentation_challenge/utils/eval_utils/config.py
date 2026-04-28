"""Configuration for evaluation pipeline."""

import logging
import os
import warnings
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

def _wrap_legacy_init(cls: type) -> type:
    """Class decorator that wraps a dataclass ``__init__`` to accept the
    deprecated ``max_instance_threads`` / ``max_semantic_threads`` keyword
    arguments and map them to ``max_workers`` for backward compatibility.

    Precedence: explicit ``max_workers`` > ``max_instance_threads`` >
    ``max_semantic_threads`` > default.
    """
    _original_init = cls.__init__

    def __init__(self, *args: Any, max_instance_threads: Any = None, max_semantic_threads: Any = None, **kwargs: Any) -> None:  # noqa: N807
        if max_instance_threads is not None:
            warnings.warn(
                "max_instance_threads is deprecated; use max_workers instead",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("max_workers", max_instance_threads)
        if max_semantic_threads is not None:
            warnings.warn(
                "max_semantic_threads is deprecated; use max_workers instead",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("max_workers", max_semantic_threads)
        _original_init(self, *args, **kwargs)

    cls.__init__ = __init__  # type: ignore[method-assign]
    return cls


@_wrap_legacy_init
@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline.

    All parameters can be set via environment variables or passed directly.
    Environment variables take precedence over defaults but not over
    explicitly passed values.

    The legacy keyword arguments ``max_instance_threads`` and
    ``max_semantic_threads`` are accepted for backward constructor
    compatibility and map to ``max_workers``.  When both ``max_workers``
    and a legacy argument are provided, ``max_workers`` takes precedence.
    """

    # Threading configuration
    max_workers: int = min(os.cpu_count() or 4, 8)
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

    @property
    def max_instance_threads(self) -> int:
        """Deprecated: use max_workers instead."""
        warnings.warn(
            "max_instance_threads is deprecated, use max_workers instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.max_workers

    @max_instance_threads.setter
    def max_instance_threads(self, value: int) -> None:
        warnings.warn(
            "max_instance_threads is deprecated, use max_workers instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.max_workers = value

    @property
    def max_semantic_threads(self) -> int:
        """Deprecated: use max_workers instead."""
        warnings.warn(
            "max_semantic_threads is deprecated, use max_workers instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.max_workers

    @max_semantic_threads.setter
    def max_semantic_threads(self, value: int) -> None:
        warnings.warn(
            "max_semantic_threads is deprecated, use max_workers instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.max_workers = value

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Load configuration from environment variables with defaults.

        ``MAX_WORKERS`` takes precedence.  When it is unset the legacy vars
        ``MAX_INSTANCE_THREADS`` and ``MAX_SEMANTIC_THREADS`` are consulted in
        that order as a fallback (each triggers a :class:`DeprecationWarning`).
        If none of the three is set the computed default is used.

        Returns:
            EvaluationConfig with values from environment or defaults.
        """
        default_workers = min(os.cpu_count() or 4, 8)
        if "MAX_WORKERS" in os.environ:
            max_workers = int(os.environ["MAX_WORKERS"])
        elif "MAX_INSTANCE_THREADS" in os.environ:
            warnings.warn(
                "MAX_INSTANCE_THREADS env var is deprecated, use MAX_WORKERS instead",
                DeprecationWarning,
                stacklevel=2,
            )
            max_workers = int(os.environ["MAX_INSTANCE_THREADS"])
        elif "MAX_SEMANTIC_THREADS" in os.environ:
            warnings.warn(
                "MAX_SEMANTIC_THREADS env var is deprecated, use MAX_WORKERS instead",
                DeprecationWarning,
                stacklevel=2,
            )
            max_workers = int(os.environ["MAX_SEMANTIC_THREADS"])
        else:
            max_workers = default_workers

        return cls(
            max_workers=max_workers,
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
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
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


# Legacy Constants (for backward compatibility during migration)
CAST_TO_NONE = [np.nan, np.inf, -np.inf, float("inf"), float("-inf")]
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(min(os.cpu_count() or 4, 8))))
MAX_INSTANCE_THREADS = MAX_WORKERS  # deprecated alias
MAX_SEMANTIC_THREADS = MAX_WORKERS  # deprecated alias
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
