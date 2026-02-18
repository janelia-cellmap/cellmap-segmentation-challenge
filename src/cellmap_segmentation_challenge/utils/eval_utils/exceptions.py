"""Custom exceptions for evaluation pipeline."""


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
