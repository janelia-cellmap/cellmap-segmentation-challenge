"""
Unit tests for refactored evaluate.py functions.

Tests cover:
- EvaluationConfig validation
- Custom exceptions
- Helper functions for match_instances
- Helper functions for score_instance
- Helper functions for score_submission
- Performance utilities
"""

import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import zarr
from upath import UPath

from cellmap_segmentation_challenge.evaluate import (
    # Configuration
    EvaluationConfig,
    # Exceptions
    EvaluationError,
    TooManyInstancesError,
    TooManyOverlapEdgesError,
    MatchingFailedError,
    ValidationError,
    # Helper functions for match_instances
    _check_instance_counts,
    _check_instance_ratio,
    _compute_instance_overlaps,
    _solve_matching_problem,
    InstanceOverlapData,
    # Helper functions for score_instance
    _compute_binary_metrics,
    _create_pathological_scores,
    _compute_hausdorff_scores,
    # Helper functions for score_submission
    _prepare_submission,
    _discover_volumes,
    _execute_parallel_scoring,
    _aggregate_and_save_results,
    # Performance utilities
    EvaluationMetrics,
    timed_operation,
    log_resource_usage,
    # Main functions
    match_instances,
    score_instance,
)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.max_instance_threads == 3
        assert config.max_semantic_threads == 25
        assert config.per_instance_threads == 25
        assert config.max_distance_cap_eps == 1e-4
        assert config.final_instance_ratio_cutoff == 10.0
        assert config.initial_instance_ratio_cutoff == 50.0
        assert config.instance_ratio_factor == 5.0
        assert config.max_overlap_edges == 5_000_000
        assert config.mcmf_cost_scale == 1_000_000

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment."""
        monkeypatch.setenv("MAX_INSTANCE_THREADS", "5")
        monkeypatch.setenv("MAX_SEMANTIC_THREADS", "30")
        monkeypatch.setenv("FINAL_INSTANCE_RATIO_CUTOFF", "15.0")

        config = EvaluationConfig.from_env()
        assert config.max_instance_threads == 5
        assert config.max_semantic_threads == 30
        assert config.final_instance_ratio_cutoff == 15.0

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = EvaluationConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_max_instance_threads(self):
        """Test validation fails for invalid max_instance_threads."""
        config = EvaluationConfig(max_instance_threads=0)
        with pytest.raises(ValueError, match="max_instance_threads must be >= 1"):
            config.validate()

    def test_validate_invalid_max_semantic_threads(self):
        """Test validation fails for invalid max_semantic_threads."""
        config = EvaluationConfig(max_semantic_threads=-1)
        with pytest.raises(ValueError, match="max_semantic_threads must be >= 1"):
            config.validate()

    def test_validate_invalid_max_distance_cap_eps(self):
        """Test validation fails for invalid max_distance_cap_eps."""
        config = EvaluationConfig(max_distance_cap_eps=0)
        with pytest.raises(ValueError, match="max_distance_cap_eps must be > 0"):
            config.validate()

    def test_validate_invalid_final_ratio_cutoff(self):
        """Test validation fails for invalid final_instance_ratio_cutoff."""
        config = EvaluationConfig(final_instance_ratio_cutoff=-1)
        with pytest.raises(ValueError, match="final_instance_ratio_cutoff must be > 0"):
            config.validate()

    def test_validate_invalid_max_overlap_edges(self):
        """Test validation fails for invalid max_overlap_edges."""
        config = EvaluationConfig(max_overlap_edges=0)
        with pytest.raises(ValueError, match="max_overlap_edges must be >= 1"):
            config.validate()


# ============================================================================
# Exception Tests
# ============================================================================


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_too_many_instances_error(self):
        """Test TooManyInstancesError attributes."""
        error = TooManyInstancesError(n_pred=100, n_gt=10, ratio=10.0, cutoff=5.0)
        assert error.n_pred == 100
        assert error.n_gt == 10
        assert error.ratio == 10.0
        assert error.cutoff == 5.0
        assert "100 predicted vs 10 ground truth" in str(error)
        assert isinstance(error, EvaluationError)

    def test_too_many_overlap_edges_error(self):
        """Test TooManyOverlapEdgesError attributes."""
        error = TooManyOverlapEdgesError(n_edges=10_000_000, max_edges=5_000_000)
        assert error.n_edges == 10_000_000
        assert error.max_edges == 5_000_000
        assert "10000000 exceeds maximum 5000000" in str(error)
        assert isinstance(error, EvaluationError)

    def test_matching_failed_error(self):
        """Test MatchingFailedError attributes."""
        error = MatchingFailedError(status=2)
        assert error.status == 2
        assert "status: 2" in str(error)
        assert isinstance(error, EvaluationError)

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Test validation error")
        assert isinstance(error, EvaluationError)


# ============================================================================
# match_instances Helper Function Tests
# ============================================================================


class TestMatchInstancesHelpers:
    """Test helper functions for match_instances."""

    def test_check_instance_counts_both_zero(self):
        """Test check_instance_counts when both are zero."""
        assert not _check_instance_counts(0, 0)

    def test_check_instance_counts_gt_zero_pred_nonzero(self):
        """Test check_instance_counts when GT is zero."""
        assert not _check_instance_counts(0, 5)

    def test_check_instance_counts_gt_nonzero_pred_zero(self):
        """Test check_instance_counts when pred is zero."""
        assert not _check_instance_counts(5, 0)

    def test_check_instance_counts_both_nonzero(self):
        """Test check_instance_counts when both are nonzero."""
        assert _check_instance_counts(5, 5)

    def test_check_instance_ratio_within_bounds(self):
        """Test ratio check passes when within bounds."""
        config = EvaluationConfig()
        # Ratio of 2:1 should be within bounds for 10 GT instances
        _check_instance_ratio(20, 10, config)  # Should not raise

    def test_check_instance_ratio_exceeds_bounds(self):
        """Test ratio check fails when exceeds bounds."""
        config = EvaluationConfig(
            final_instance_ratio_cutoff=2.0, initial_instance_ratio_cutoff=5.0
        )
        # Ratio of 100:1 should exceed bounds
        with pytest.raises(TooManyInstancesError):
            _check_instance_ratio(100, 1, config)

    def test_compute_instance_overlaps_no_overlaps(self):
        """Test overlap computation with no overlaps."""
        # GT has instance 1 in top-left, pred has instance 1 in bottom-right (no overlap)
        gt = np.array([[1, 1], [0, 0]])
        pred = np.array([[0, 0], [1, 1]])

        overlap_data = _compute_instance_overlaps(gt, pred, nG=1, nP=1, max_edges=1000)

        assert overlap_data.nG == 1
        assert overlap_data.nP == 1
        assert len(overlap_data.rows) == 0
        assert len(overlap_data.cols) == 0
        assert len(overlap_data.iou_vals) == 0

    def test_compute_instance_overlaps_with_overlaps(self):
        """Test overlap computation with overlaps."""
        gt = np.array([[1, 1], [0, 0]])
        pred = np.array([[1, 1], [0, 0]])

        overlap_data = _compute_instance_overlaps(gt, pred, nG=1, nP=1, max_edges=1000)

        assert overlap_data.nG == 1
        assert overlap_data.nP == 1
        assert len(overlap_data.rows) == 1
        assert len(overlap_data.cols) == 1
        assert overlap_data.iou_vals[0] == pytest.approx(1.0)  # Perfect overlap

    def test_compute_instance_overlaps_too_many_edges(self):
        """Test overlap computation fails with too many edges."""
        # Create many small instances to exceed edge limit
        gt = np.arange(100).reshape(10, 10)
        pred = np.arange(100).reshape(10, 10)

        with pytest.raises(TooManyOverlapEdgesError):
            _compute_instance_overlaps(gt, pred, nG=100, nP=100, max_edges=10)

    def test_solve_matching_problem_simple(self):
        """Test min-cost flow matching with simple case."""
        # Create overlap data for perfect match
        overlap_data = InstanceOverlapData(
            nG=2,
            nP=2,
            rows=np.array([0, 1]),
            cols=np.array([0, 1]),
            iou_vals=np.array([1.0, 1.0], dtype=np.float32),
        )

        mapping = _solve_matching_problem(overlap_data, cost_scale=1000000)

        # Should map pred 1->gt 1 and pred 2->gt 2
        assert mapping[1] == 1
        assert mapping[2] == 2


# ============================================================================
# score_instance Helper Function Tests
# ============================================================================


class TestScoreInstanceHelpers:
    """Test helper functions for score_instance."""

    def test_compute_binary_metrics_perfect_match(self):
        """Test binary metrics with perfect match."""
        truth = np.array([[1, 1], [0, 0]])
        pred = np.array([[1, 1], [0, 0]])

        metrics = _compute_binary_metrics(truth, pred)

        assert metrics["iou"] == pytest.approx(1.0)
        assert metrics["dice_score"] == pytest.approx(1.0)
        assert metrics["binary_accuracy"] == pytest.approx(1.0)

    def test_compute_binary_metrics_no_overlap(self):
        """Test binary metrics with no overlap."""
        truth = np.array([[1, 1], [0, 0]])
        pred = np.array([[0, 0], [1, 1]])

        metrics = _compute_binary_metrics(truth, pred)

        assert metrics["iou"] == 0.0
        assert metrics["binary_accuracy"] == 0.0

    def test_create_pathological_scores(self):
        """Test creation of pathological scores."""
        binary_metrics = {"iou": 0.5, "dice_score": 0.6, "binary_accuracy": 0.7}
        voi_metrics = {"voi_split": 0.1, "voi_merge": 0.2}

        scores = _create_pathological_scores(
            binary_metrics,
            voi_metrics,
            hausdorff_distance_max=100.0,
            voxel_size=(4.0, 4.0, 4.0),
            status="test_failure",
        )

        assert scores["mean_accuracy"] == 0
        assert scores["combined_score"] == 0
        assert scores["hausdorff_distance"] == 100.0
        assert scores["iou"] == 0.5
        assert scores["status"] == "test_failure"
        assert scores["voi_split"] == 0.1

    def test_compute_hausdorff_scores_only_background(self):
        """Test Hausdorff score computation with only background."""
        mapping = {0: 0}
        truth = np.zeros((10, 10))
        pred = np.zeros((10, 10))

        distances = _compute_hausdorff_scores(
            mapping,
            truth,
            pred,
            n_pred=0,
            voxel_size=(4.0, 4.0),
            hausdorff_distance_max=100.0,
        )

        assert distances == [pytest.approx(0.0)]

    def test_compute_hausdorff_scores_no_mapping(self):
        """Test Hausdorff score computation with no mapping."""
        mapping = {}
        truth = np.zeros((10, 10))
        pred = np.zeros((10, 10))

        distances = _compute_hausdorff_scores(
            mapping,
            truth,
            pred,
            n_pred=5,
            voxel_size=(4.0, 4.0),
            hausdorff_distance_max=100.0,
        )

        assert distances == [100.0]


# ============================================================================
# score_submission Helper Function Tests
# ============================================================================


class TestScoreSubmissionHelpers:
    """Test helper functions for score_submission."""

    @patch("cellmap_segmentation_challenge.evaluate.unzip_file")
    @patch("cellmap_segmentation_challenge.evaluate.ensure_valid_submission")
    def test_prepare_submission(self, mock_ensure_valid, mock_unzip):
        """Test submission preparation."""
        mock_unzip.return_value = UPath("/tmp/submission.zarr")

        result = _prepare_submission("/tmp/submission.zip")

        mock_unzip.assert_called_once_with("/tmp/submission.zip")
        mock_ensure_valid.assert_called_once()
        assert isinstance(result, UPath)

    def test_discover_volumes_matching(self, tmp_path):
        """Test volume discovery with matching volumes."""
        # Create test directories
        submission_path = tmp_path / "submission"
        truth_path = tmp_path / "truth"
        submission_path.mkdir()
        truth_path.mkdir()

        (submission_path / "crop1").mkdir()
        (submission_path / "crop2").mkdir()
        (truth_path / "crop1").mkdir()
        (truth_path / "crop2").mkdir()
        (truth_path / "crop3").mkdir()

        found, missing = _discover_volumes(UPath(submission_path), UPath(truth_path))

        assert set(found) == {"crop1", "crop2"}
        assert set(missing) == {"crop3"}

    def test_discover_volumes_no_matches(self, tmp_path):
        """Test volume discovery with no matches raises error."""
        submission_path = tmp_path / "submission"
        truth_path = tmp_path / "truth"
        submission_path.mkdir()
        truth_path.mkdir()

        (submission_path / "wrong1").mkdir()
        (truth_path / "crop1").mkdir()

        with pytest.raises(ValueError, match="No volumes found to score"):
            _discover_volumes(UPath(submission_path), UPath(truth_path))

    def test_aggregate_and_save_results(self, tmp_path):
        """Test result aggregation and saving."""
        result_file = str(tmp_path / "results.json")
        config = EvaluationConfig()

        results = [
            ("crop1", "label1", {"accuracy": 0.9, "status": "scored"}),
            ("crop1", "label2", {"iou": 0.8, "status": "scored"}),
        ]
        missing_scores = {}

        with patch(
            "cellmap_segmentation_challenge.evaluate.update_scores"
        ) as mock_update:
            mock_update.return_value = (
                {
                    "overall_score": 0.85,
                    "overall_instance_score": 0.9,
                    "overall_semantic_score": 0.8,
                },
                {
                    "overall_score": 0.85,
                    "overall_instance_score": 0.9,
                    "overall_semantic_score": 0.8,
                },
            )

            scores = _aggregate_and_save_results(
                results, missing_scores, result_file, config
            )

            assert scores["overall_score"] == 0.85
            mock_update.assert_called_once()


# ============================================================================
# Performance Utility Tests
# ============================================================================


class TestPerformanceUtilities:
    """Test performance monitoring utilities."""

    def test_evaluation_metrics_to_dict(self):
        """Test EvaluationMetrics to_dict conversion."""
        metrics = EvaluationMetrics(
            total_evaluations=10,
            successful_evaluations=8,
            failed_evaluations=2,
            total_duration_seconds=100.0,
            avg_instance_score=0.9,
            avg_semantic_score=0.85,
        )

        result = metrics.to_dict()

        assert result["total_evaluations"] == 10
        assert result["successful_evaluations"] == 8
        assert result["success_rate"] == 0.8
        assert result["avg_duration_seconds"] == 10.0
        assert result["avg_instance_score"] == 0.9

    def test_timed_operation_decorator_success(self):
        """Test timed_operation decorator on successful function."""

        @timed_operation("test_op")
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_timed_operation_decorator_failure(self):
        """Test timed_operation decorator on failing function."""

        @timed_operation("test_op")
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_func()

    def test_log_resource_usage(self):
        """Test log_resource_usage function."""
        # Should not raise
        log_resource_usage("test_operation", crop="crop1", label="label1")


# ============================================================================
# Integration Tests with New Signatures
# ============================================================================


class TestRefactoredIntegration:
    """Integration tests for refactored functions."""

    def test_match_instances_with_config(self):
        """Test match_instances with custom config."""
        gt = np.array([[1, 1], [0, 0]])
        pred = np.array([[1, 1], [0, 0]])
        config = EvaluationConfig(mcmf_cost_scale=500000)

        mapping = match_instances(gt, pred, config=config)

        assert 1 in mapping
        assert mapping[1] == 1

    def test_match_instances_raises_too_many_instances(self):
        """Test match_instances raises TooManyInstancesError."""
        # Create GT with 1 instance, pred with many (same shape)
        gt = np.ones((10, 10), dtype=int)  # All pixels are instance 1
        pred = np.arange(100).reshape(10, 10) + 1  # 100 instances

        config = EvaluationConfig(final_instance_ratio_cutoff=2.0)

        with pytest.raises(TooManyInstancesError):
            match_instances(gt, pred, config=config)

    def test_score_instance_with_config(self):
        """Test score_instance with custom config."""
        pred = np.array([[1, 1], [0, 0]])
        truth = np.array([[1, 1], [0, 0]])
        voxel_size = (4.0, 4.0)

        config = EvaluationConfig()
        scores = score_instance(pred, truth, voxel_size, config=config)

        assert "mean_accuracy" in scores
        assert "combined_score" in scores
        assert scores["status"] == "scored"

    def test_score_instance_handles_too_many_instances(self):
        """Test score_instance handles TooManyInstancesError gracefully."""
        # Create scenario that triggers TooManyInstancesError (same shape)
        truth = np.ones((10, 10), dtype=int)  # All pixels are instance 1
        pred = np.arange(100).reshape(10, 10) + 1  # 100 instances
        voxel_size = (4.0, 4.0)

        config = EvaluationConfig(final_instance_ratio_cutoff=2.0)
        scores = score_instance(pred, truth, voxel_size, config=config)

        assert scores["status"] == "skipped_too_many_instances"
        assert scores["mean_accuracy"] == 0
        assert scores["combined_score"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
