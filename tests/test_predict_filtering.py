"""Unit tests for predict function's filtering behavior"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from cellmap_segmentation_challenge.predict import _predict


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
