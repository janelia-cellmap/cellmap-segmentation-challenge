"""Unit tests for utility functions in cellmap_segmentation_challenge.utils"""

import pytest
import tempfile
import os

from cellmap_segmentation_challenge.utils.utils import (
    format_coordinates,
    format_string,
    download_file,
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

    def test_download_file_success(self):
        """Test successful file download from a real URL"""
        # Use a small, reliable test file from the repository itself
        url = "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/LICENSE"

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            dest_path = f.name

        try:
            download_file(url, dest_path)

            # Verify the file was downloaded and has content
            assert os.path.exists(dest_path)
            with open(dest_path, "r") as f:
                content = f.read()
                assert len(content) > 0
                # LICENSE file should contain "MIT"
                assert "MIT" in content or "License" in content
        finally:
            if os.path.exists(dest_path):
                os.unlink(dest_path)

    def test_download_file_invalid_url(self):
        """Test file download with invalid URL"""
        import requests

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            dest_path = f.name

        try:
            with pytest.raises(requests.exceptions.RequestException):
                download_file(
                    "http://invalid-url-that-does-not-exist-12345.com/file.txt",
                    dest_path,
                )
        finally:
            if os.path.exists(dest_path):
                os.unlink(dest_path)
