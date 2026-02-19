"""Unit tests for utility functions in cellmap_segmentation_challenge.utils"""

import os
import tempfile

import pytest
import torch

from cellmap_segmentation_challenge.utils.utils import (
    download_file,
    format_coordinates,
    format_string,
    get_data_from_batch,
    get_singleton_dim,
    squeeze_singleton_dim,
    structure_model_output,
    unsqueeze_singleton_dim,
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


class TestGetSingletonDim:
    def test_no_singleton(self):
        assert get_singleton_dim((4, 64, 64)) is None

    def test_first_dim_singleton(self):
        assert get_singleton_dim((1, 64, 64)) == 0

    def test_middle_dim_singleton(self):
        assert get_singleton_dim((4, 1, 64)) == 1

    def test_last_dim_singleton(self):
        assert get_singleton_dim((4, 64, 1)) == 2

    def test_multiple_singletons_returns_first(self):
        assert get_singleton_dim((1, 1, 64)) == 0

    def test_all_singletons(self):
        assert get_singleton_dim((1, 1, 1)) == 0


class TestSqueezeUnsqueezeSingletonDim:
    def test_squeeze_tensor(self):
        t = torch.zeros(2, 1, 8, 8)
        result = squeeze_singleton_dim(t, dim=1)
        assert result.shape == (2, 8, 8)

    def test_squeeze_dict(self):
        d = {"a": torch.zeros(2, 1, 8, 8), "b": torch.zeros(2, 1, 4, 4)}
        result = squeeze_singleton_dim(d, dim=1)
        assert result["a"].shape == (2, 8, 8)
        assert result["b"].shape == (2, 4, 4)

    def test_unsqueeze_tensor(self):
        t = torch.zeros(2, 8, 8)
        result = unsqueeze_singleton_dim(t, dim=1)
        assert result.shape == (2, 1, 8, 8)

    def test_unsqueeze_dict(self):
        d = {"a": torch.zeros(2, 8, 8), "b": torch.zeros(2, 4, 4)}
        result = unsqueeze_singleton_dim(d, dim=1)
        assert result["a"].shape == (2, 1, 8, 8)
        assert result["b"].shape == (2, 1, 4, 4)

    def test_squeeze_unsqueeze_roundtrip_tensor(self):
        t = torch.zeros(2, 1, 8, 8)
        assert (
            unsqueeze_singleton_dim(squeeze_singleton_dim(t, dim=1), dim=1).shape
            == t.shape
        )

    def test_squeeze_unsqueeze_roundtrip_dict(self):
        d = {"x": torch.zeros(2, 1, 8, 8)}
        result = unsqueeze_singleton_dim(squeeze_singleton_dim(d, dim=1), dim=1)
        assert result["x"].shape == d["x"].shape


class TestGetDataFromBatch:
    def test_single_key_returns_tensor(self):
        batch = {"input": torch.zeros(2, 1, 8, 8)}
        result = get_data_from_batch(batch, ["input"], "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 1, 8, 8)

    def test_multiple_keys_returns_dict(self):
        batch = {"a": torch.zeros(2, 1, 8, 8), "b": torch.zeros(2, 3, 8, 8)}
        result = get_data_from_batch(batch, ["a", "b"], "cpu")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        assert result["a"].shape == (2, 1, 8, 8)
        assert result["b"].shape == (2, 3, 8, 8)


CLASSES = ["nuc", "er", "mito"]


class TestStructureModelOutput:
    # --- tensor inputs ---

    def test_tensor_one_channel_per_class(self):
        out = torch.zeros(2, 3, 8, 8)  # 3 classes, 1 ch each
        result = structure_model_output(out, CLASSES)
        assert set(result.keys()) == {"output"}
        assert isinstance(result["output"], torch.Tensor)
        assert result["output"].shape == (2, 3, 8, 8)

    def test_tensor_multi_channel_per_class(self):
        out = torch.zeros(2, 6, 8, 8)  # 3 classes, 2 ch each
        result = structure_model_output(out, CLASSES, num_channels_per_class=2)
        assert set(result.keys()) == {"output"}
        assert set(result["output"].keys()) == set(CLASSES)
        for v in result["output"].values():
            assert v.shape == (2, 2, 8, 8)

    def test_tensor_channel_slices_are_correct(self):
        # Fill each class's channels with the class index to verify slicing
        out = torch.cat([torch.full((2, 2, 4, 4), float(i)) for i in range(3)], dim=1)
        result = structure_model_output(out, CLASSES, num_channels_per_class=2)
        for i, cls in enumerate(CLASSES):
            assert result["output"][cls].eq(float(i)).all()

    def test_tensor_wrong_channel_count_raises(self):
        out = torch.zeros(2, 5, 8, 8)  # not divisible by 3
        with pytest.raises(ValueError, match="does not match"):
            structure_model_output(out, CLASSES)

    # --- dict inputs: keys == classes ---

    def test_class_key_dict_wrapped(self):
        d = {cls: torch.zeros(2, 2, 8, 8) for cls in CLASSES}
        result = structure_model_output(d, CLASSES)
        assert set(result.keys()) == {"output"}
        assert set(result["output"].keys()) == set(CLASSES)

    def test_class_key_dict_values_unchanged(self):
        tensors = {
            cls: torch.full((2, 2, 8, 8), float(i)) for i, cls in enumerate(CLASSES)
        }
        result = structure_model_output(tensors, CLASSES)
        for cls, t in tensors.items():
            assert result["output"][cls].data_ptr() == t.data_ptr()

    # --- dict inputs: keys != classes (e.g. resolution levels) ---

    def test_resolution_dict_one_channel_per_class(self):
        d = {"8nm": torch.zeros(2, 3, 8, 8), "32nm": torch.zeros(2, 3, 4, 4)}
        result = structure_model_output(d, CLASSES)
        assert set(result.keys()) == {"8nm", "32nm"}
        # one channel per class → value is a plain tensor, not split into sub-dict
        assert isinstance(result["8nm"], torch.Tensor)

    def test_resolution_dict_multi_channel_per_class(self):
        d = {"8nm": torch.zeros(2, 6, 8, 8), "32nm": torch.zeros(2, 6, 4, 4)}
        result = structure_model_output(d, CLASSES, num_channels_per_class=2)
        for key in ("8nm", "32nm"):
            assert set(result[key].keys()) == set(CLASSES)
            for v in result[key].values():
                assert v.shape[1] == 2

    def test_resolution_dict_wrong_channels_raises(self):
        d = {"8nm": torch.zeros(2, 5, 8, 8)}
        with pytest.raises(ValueError, match="does not match"):
            structure_model_output(d, CLASSES)

    def test_tensor_multi_channel_wrong_count_raises(self):
        """Test that tensor with wrong channel count for num_channels_per_class raises error"""
        out = torch.zeros(2, 5, 8, 8)  # 5 channels != 3 classes × 2 channels
        with pytest.raises(ValueError, match="does not match expected"):
            structure_model_output(out, CLASSES, num_channels_per_class=2)

    def test_resolution_dict_multi_channel_wrong_count_raises(self):
        """Test that dict values with wrong channel count for num_channels_per_class raises error"""
        d = {"8nm": torch.zeros(2, 5, 8, 8)}  # 5 channels != 3 classes × 2 channels
        with pytest.raises(ValueError, match="does not match expected"):
            structure_model_output(d, CLASSES, num_channels_per_class=2)


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
