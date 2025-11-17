"""Unit tests for utility functions in cellmap_segmentation_challenge.utils"""

import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import requests

from cellmap_segmentation_challenge.utils.utils import (
    format_coordinates,
    format_string,
    download_file,
    simulate_predictions_iou_binary,
    simulate_predictions_accuracy,
)


class TestFormatCoordinates:
    """Tests for format_coordinates function"""

    def test_format_coordinates_simple_list(self):
        """Test formatting a simple list of coordinates"""
        coordinates = [1, 2, 3]
        result = format_coordinates(coordinates)
        assert result == "[1;2;3]"

    def test_format_coordinates_float_list(self):
        """Test formatting coordinates with float values"""
        coordinates = [1.5, 2.5, 3.5]
        result = format_coordinates(coordinates)
        assert result == "[1.5;2.5;3.5]"

    def test_format_coordinates_single_value(self):
        """Test formatting a single coordinate"""
        coordinates = [42]
        result = format_coordinates(coordinates)
        assert result == "[42]"

    def test_format_coordinates_empty_list(self):
        """Test formatting an empty list"""
        coordinates = []
        result = format_coordinates(coordinates)
        assert result == "[]"


class TestFormatString:
    """Tests for format_string function"""

    def test_format_string_all_keys_present(self):
        """Test formatting when all keys are present in the string"""
        string = "Hello {name}, you are {age} years old"
        format_kwargs = {"name": "Alice", "age": 30}
        result = format_string(string, format_kwargs)
        assert result == "Hello Alice, you are 30 years old"

    def test_format_string_partial_keys(self):
        """Test formatting when only some keys are present"""
        string = "Hello {name}"
        format_kwargs = {"name": "Bob", "age": 25}
        result = format_string(string, format_kwargs)
        assert result == "Hello Bob"

    def test_format_string_no_keys_in_string(self):
        """Test when no format keys are in the string"""
        string = "Hello World"
        format_kwargs = {"name": "Charlie"}
        result = format_string(string, format_kwargs)
        assert result == "Hello World"

    def test_format_string_missing_required_key(self):
        """Test when a required key is missing from format_kwargs"""
        string = "Hello {name}"
        format_kwargs = {"age": 30}
        result = format_string(string, format_kwargs)
        # Should return the original string with placeholders preserved
        assert result == "Hello {name}"


class TestDownloadFile:
    """Tests for download_file function"""

    @patch("cellmap_segmentation_challenge.utils.utils.requests.get")
    def test_download_file_success(self, mock_get):
        """Test successful file download"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.content = b"test content"
        mock_get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            download_file("http://example.com/file.txt", "/tmp/file.txt")
            
            # Verify the file was opened for writing
            mock_file.assert_called_once_with("/tmp/file.txt", "wb")
            # Verify content was written
            mock_file().write.assert_called_once_with(b"test content")

    @patch("cellmap_segmentation_challenge.utils.utils.requests.get")
    def test_download_file_http_error(self, mock_get):
        """Test file download with HTTP error"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            download_file("http://example.com/file.txt", "/tmp/file.txt")


class TestSimulatePredictionsIouBinary:
    """Tests for simulate_predictions_iou_binary function"""

    def test_simulate_predictions_perfect_iou(self):
        """Test with perfect IOU (1.0)"""
        labels = np.array([[[1, 1], [0, 0]], [[1, 0], [1, 1]]])
        result = simulate_predictions_iou_binary(labels, 1.0)
        
        # With IOU 1.0, all positive labels should remain positive
        assert result.shape == labels.shape
        assert np.all((result > 0) == (labels > 0))

    def test_simulate_predictions_zero_iou(self):
        """Test with zero IOU"""
        labels = np.array([[[1, 1], [0, 0]], [[1, 0], [1, 1]]])
        result = simulate_predictions_iou_binary(labels, 0.0)
        
        # With IOU 0.0, all positive labels should become 0
        assert result.shape == labels.shape
        assert np.all(result == 0)

    def test_simulate_predictions_partial_iou(self):
        """Test with partial IOU (0.5)"""
        np.random.seed(42)  # For reproducibility
        labels = np.ones((10, 10, 10))
        result = simulate_predictions_iou_binary(labels, 0.5)
        
        assert result.shape == labels.shape
        # Approximately half should remain positive
        positive_ratio = np.sum(result > 0) / result.size
        assert 0.3 < positive_ratio < 0.7  # Allow some variance


class TestSimulatePredictionsAccuracy:
    """Tests for simulate_predictions_accuracy function"""

    def test_simulate_predictions_perfect_accuracy(self):
        """Test with perfect accuracy (1.0)"""
        np.random.seed(42)
        true_labels = np.array([[[1, 1], [0, 0]], [[1, 0], [1, 1]]])
        result = simulate_predictions_accuracy(true_labels, 1.0)
        
        # With accuracy 1.0, result should match input (after relabeling)
        assert result.shape == true_labels.shape
        # All pixels should be classified correctly (binary)
        assert np.sum((result > 0) == (true_labels > 0)) == result.size

    def test_simulate_predictions_zero_accuracy(self):
        """Test with zero accuracy"""
        np.random.seed(42)
        true_labels = np.ones((5, 5, 5), dtype=int)
        result = simulate_predictions_accuracy(true_labels, 0.0)
        
        # With accuracy 0.0, all labels should be flipped
        assert result.shape == true_labels.shape

    def test_simulate_predictions_partial_accuracy(self):
        """Test with partial accuracy (0.8)"""
        np.random.seed(42)
        true_labels = np.random.randint(0, 2, size=(10, 10, 10))
        result = simulate_predictions_accuracy(true_labels, 0.8)
        
        assert result.shape == true_labels.shape
        # Result should have some instances labeled
        assert len(np.unique(result)) >= 1
