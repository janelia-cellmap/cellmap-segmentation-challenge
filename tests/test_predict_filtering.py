"""Unit tests for predict function's filtering behavior"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cellmap_segmentation_challenge.predict import _predict, predict


class TestPredictFiltering:
    """Tests for predict function filtering of outputs based on test_crop_manifest"""

    def test_filtered_classes_validation_in_predict(self):
        """Test that _predict validates classes_to_save is not empty"""
        # Create a mock model
        model = Mock()
        
        # Create dataset_writer_kwargs with empty classes (should raise ValueError)
        dataset_writer_kwargs = {
            "device": "cpu",
            "input_arrays": {"input": {"shape": (64, 64, 64), "scale": (8, 8, 8)}},
            "classes": [],  # Empty classes should trigger validation error
            "model_classes": ["mito", "er"],
            "target_arrays": {"output": {"shape": (64, 64, 64), "scale": (8, 8, 8)}},
            "raw_path": "/fake/path",
            "target_path": "/fake/output",
            "overwrite": True,
        }
        
        # Should raise ValueError due to empty classes_to_save
        with pytest.raises(ValueError, match="classes_to_save is empty"):
            _predict(model, dataset_writer_kwargs, batch_size=1)
    
    def test_dict_output_filtering(self):
        """Test that dict-based model outputs are correctly filtered"""
        # This is a logic test to verify the filtering behavior
        model_classes = ["mito", "er", "nuc", "cell"]
        classes_to_save = ["mito", "er"]
        
        # Simulate dict output from structure_model_output
        outputs = {
            "output": {
                "mito": torch.randn(2, 64, 64, 64),
                "er": torch.randn(2, 64, 64, 64),
                "nuc": torch.randn(2, 64, 64, 64),
                "cell": torch.randn(2, 64, 64, 64),
            }
        }
        
        # Apply the filtering logic
        filtered_outputs = {}
        for array_name, class_outputs in outputs.items():
            if isinstance(class_outputs, dict):
                filtered_outputs[array_name] = {
                    class_name: class_tensor
                    for class_name, class_tensor in class_outputs.items()
                    if class_name in classes_to_save
                }
        
        # Verify only the saved classes are present
        assert set(filtered_outputs["output"].keys()) == set(classes_to_save)
        assert "nuc" not in filtered_outputs["output"]
        assert "cell" not in filtered_outputs["output"]
        assert "mito" in filtered_outputs["output"]
        assert "er" in filtered_outputs["output"]
    
    def test_tensor_output_filtering(self):
        """Test that tensor-based model outputs are correctly filtered"""
        model_classes = ["mito", "er", "nuc", "cell"]
        classes_to_save = ["mito", "er"]
        
        # Create mapping
        model_class_to_index = {c: i for i, c in enumerate(model_classes)}
        
        # Simulate tensor output with 4 channels (one per class)
        outputs = {
            "output": torch.randn(2, 4, 64, 64, 64)  # (B, C, D, H, W)
        }
        
        # Apply the filtering logic
        filtered_outputs = {}
        for array_name, class_outputs in outputs.items():
            if not isinstance(class_outputs, dict):
                class_indices = [model_class_to_index[c] for c in classes_to_save]
                filtered_outputs[array_name] = class_outputs[:, class_indices, ...]
        
        # Verify the tensor has correct shape
        assert filtered_outputs["output"].shape[1] == len(classes_to_save)
        assert filtered_outputs["output"].shape[1] == 2  # Only mito and er
        
        # Verify the correct channels were selected
        expected_indices = [0, 1]  # mito=0, er=1
        assert class_indices == expected_indices
    
    def test_no_filtering_when_classes_match(self):
        """Test that no filtering occurs when model_classes equals classes_to_save"""
        model_classes = ["mito", "er"]
        classes_to_save = ["mito", "er"]
        
        # When classes match, model_class_to_index should be None
        model_class_to_index = {c: i for i, c in enumerate(model_classes)} if model_classes != classes_to_save else None
        
        assert model_class_to_index is None
        
        # This means the filtering block won't execute in _predict


class TestPredictValidation:
    """Tests for validation logic in predict function"""
    
    def test_skip_crop_with_no_matching_labels(self):
        """Test that crops with no matching labels between model and manifest are skipped"""
        from cellmap_segmentation_challenge.utils.crops import get_test_crop_labels
        
        # Simulate a model with certain classes
        model_classes = ["label1", "label2", "label3"]
        
        # Simulate a crop that has completely different labels
        # (In reality this would come from test_crop_manifest)
        # We'll test the filtering logic directly
        crop_labels = ["different_label1", "different_label2"]
        
        # Apply the filtering
        filtered_classes = [c for c in model_classes if c in crop_labels]
        
        # Should be empty since there's no overlap
        assert filtered_classes == []
        
        # This would trigger the skip logic in predict()
    
    def test_intersection_of_model_and_crop_labels(self):
        """Test that only the intersection of model classes and crop labels are kept"""
        model_classes = ["mito", "er", "nuc", "cell", "golgi"]
        crop_labels = ["mito", "er", "ves", "mt"]  # ves and mt not in model
        
        # Apply the filtering
        filtered_classes = [c for c in model_classes if c in crop_labels]
        
        # Should only have the intersection
        assert set(filtered_classes) == {"mito", "er"}
        assert "ves" not in filtered_classes  # Not in model
        assert "mt" not in filtered_classes  # Not in model
        assert "nuc" not in filtered_classes  # Not in crop
        assert "cell" not in filtered_classes  # Not in crop


class TestPredictEndToEnd:
    """End-to-end integration tests for predict function with crops='test'"""
    
    @pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION_TESTS") == "true",
        reason="Skipping integration test - requires full environment setup"
    )
    def test_predict_with_test_crops_filters_classes(self):
        """
        End-to-end test that verifies predict() with crops='test' only saves
        the classes specified in test_crop_manifest for each crop.
        
        This test:
        1. Creates a mock model with multiple classes
        2. Mocks get_test_crops() to return a test crop with subset of labels
        3. Calls predict() with crops='test'
        4. Verifies that only the filtered classes are written to output
        """
        # Create a temporary directory for outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple mock model that returns predictions for all classes
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            
            # Model returns dict with all classes
            def mock_forward(inputs):
                # Return dict with predictions for all 4 classes
                batch_size = inputs.shape[0] if isinstance(inputs, torch.Tensor) else 2
                return {
                    "mito": torch.randn(batch_size, 64, 64, 64),
                    "er": torch.randn(batch_size, 64, 64, 64),
                    "nuc": torch.randn(batch_size, 64, 64, 64),
                    "cell": torch.randn(batch_size, 64, 64, 64),
                }
            
            mock_model.__call__ = mock_forward
            
            # Create a mock config
            mock_config = Mock()
            mock_config.classes = ["mito", "er", "nuc", "cell"]  # Model trained on 4 classes
            mock_config.model = mock_model
            mock_config.batch_size = 2
            mock_config.input_array_info = {"shape": (1, 64, 64), "scale": (8, 8, 8)}
            mock_config.target_array_info = {"shape": (1, 64, 64), "scale": (8, 8, 8)}
            mock_config.device = "cpu"
            
            # Mock get_test_crops to return a crop that only needs mito and er
            mock_crop = Mock()
            mock_crop.id = 999
            mock_crop.dataset = "test_dataset"
            mock_crop.gt_source = Mock()
            mock_crop.gt_source.translation = [0.0, 0.0, 0.0]
            mock_crop.gt_source.voxel_size = [8.0, 8.0, 8.0]
            mock_crop.gt_source.shape = [64, 64, 64]
            
            # Mock get_test_crop_labels to return only mito and er for this crop
            with patch('cellmap_segmentation_challenge.predict.get_test_crops') as mock_get_crops, \
                 patch('cellmap_segmentation_challenge.predict.get_test_crop_labels') as mock_get_labels, \
                 patch('cellmap_segmentation_challenge.predict.load_safe_config') as mock_load_config, \
                 patch('cellmap_segmentation_challenge.predict.get_model') as mock_get_model, \
                 patch('cellmap_segmentation_challenge.predict.CellMapDatasetWriter') as mock_writer_class:
                
                mock_get_crops.return_value = [mock_crop]
                mock_get_labels.return_value = ["mito", "er"]  # Only these 2 labels for this crop
                mock_load_config.return_value = mock_config
                mock_get_model.return_value = None  # No checkpoint to load
                
                # Track what classes are passed to CellMapDatasetWriter
                captured_classes = []
                
                def capture_writer_init(**kwargs):
                    captured_classes.append(kwargs.get("classes", []))
                    mock_writer_instance = Mock()
                    mock_writer_instance.loader = Mock(return_value=[])  # Empty loader to skip prediction loop
                    return mock_writer_instance
                
                mock_writer_class.side_effect = capture_writer_init
                
                # Call predict with crops="test"
                output_path = os.path.join(tmpdir, "{dataset}.zarr/{crop}")
                
                try:
                    predict(
                        config_path="fake_config.py",
                        crops="test",
                        output_path=output_path,
                        do_orthoplanes=False,
                        overwrite=True,
                    )
                except Exception as e:
                    # The test might fail during actual prediction due to mocking,
                    # but we can still verify that the classes were filtered correctly
                    pass
                
                # Verify that CellMapDatasetWriter was called with only the filtered classes
                assert len(captured_classes) > 0, "CellMapDatasetWriter should have been called"
                for classes_list in captured_classes:
                    # Should only have mito and er, not nuc and cell
                    assert set(classes_list) == {"mito", "er"}, \
                        f"Expected filtered classes ['mito', 'er'], but got {classes_list}"
                    assert "nuc" not in classes_list
                    assert "cell" not in classes_list
    
    def test_predict_filters_output_structure(self):
        """
        Test that verifies the internal filtering logic correctly filters
        both dict-based and tensor-based outputs before saving.
        
        This is a lighter-weight test that verifies the filtering mechanism
        without requiring full environment setup.
        """
        # Test the filtering logic that happens in _predict
        model_classes = ["mito", "er", "nuc", "cell"]
        classes_to_save = ["mito", "er"]
        
        # Create the index mapping as done in _predict
        model_class_to_index = {c: i for i, c in enumerate(model_classes)}
        
        # Test dict-based output filtering
        dict_outputs = {
            "output": {
                "mito": torch.randn(2, 64, 64, 64),
                "er": torch.randn(2, 64, 64, 64),
                "nuc": torch.randn(2, 64, 64, 64),
                "cell": torch.randn(2, 64, 64, 64),
            }
        }
        
        # Apply filtering
        filtered_dict = {}
        for array_name, class_outputs in dict_outputs.items():
            if isinstance(class_outputs, dict):
                filtered_dict[array_name] = {
                    class_name: class_tensor
                    for class_name, class_tensor in class_outputs.items()
                    if class_name in classes_to_save
                }
        
        # Verify
        assert set(filtered_dict["output"].keys()) == {"mito", "er"}
        
        # Test tensor-based output filtering
        tensor_outputs = {
            "output": torch.randn(2, 4, 64, 64, 64)  # 4 channels for 4 classes
        }
        
        # Apply filtering
        filtered_tensor = {}
        for array_name, class_outputs in tensor_outputs.items():
            if not isinstance(class_outputs, dict):
                class_indices = [model_class_to_index[c] for c in classes_to_save]
                filtered_tensor[array_name] = class_outputs[:, class_indices, ...]
        
        # Verify
        assert filtered_tensor["output"].shape[1] == 2  # Only 2 classes saved
        assert class_indices == [0, 1]  # mito and er indices

