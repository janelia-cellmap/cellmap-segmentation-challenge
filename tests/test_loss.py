"""Unit tests for loss functions in cellmap_segmentation_challenge.utils.loss"""

import torch

from cellmap_segmentation_challenge.utils.loss import CellMapLossWrapper


class TestCellMapLossWrapper:
    """Tests for CellMapLossWrapper class"""

    def test_init_with_mse_loss(self):
        """Test initialization with MSE loss"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        assert isinstance(loss_wrapper.loss_fn, torch.nn.MSELoss)
        assert loss_wrapper.kwargs["reduction"] == "none"

    def test_init_with_bce_loss(self):
        """Test initialization with BCE loss"""
        loss_wrapper = CellMapLossWrapper(torch.nn.BCELoss)
        assert isinstance(loss_wrapper.loss_fn, torch.nn.BCELoss)

    def test_calc_loss_no_nans(self):
        """Test calc_loss with no NaN values"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        loss = loss_wrapper.calc_loss(outputs, targets)

        # With identical outputs and targets, MSE should be 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_calc_loss_with_nans(self):
        """Test calc_loss with NaN values in targets"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])

        loss = loss_wrapper.calc_loss(outputs, targets)

        # Loss should be computed only for non-NaN values
        # Expected loss = ((1-1)^2 + (3-3)^2 + (4-4)^2) / 3 = 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_calc_loss_all_nans(self):
        """Test calc_loss when all targets are NaN"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor(
            [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )

        loss = loss_wrapper.calc_loss(outputs, targets)

        # When all targets are NaN, the loss is 0 (no valid pixels to compute loss on)
        assert torch.allclose(loss, torch.tensor(0.0)) or torch.isnan(loss)

    def test_forward_tensor_inputs(self):
        """Test forward with tensor inputs"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

        loss = loss_wrapper.forward(outputs, targets)

        # MSE = ((1-1.5)^2 + (2-2.5)^2 + (3-3.5)^2 + (4-4.5)^2) / 4 = 0.25
        assert torch.allclose(loss, torch.tensor(0.25), atol=1e-6)

    def test_forward_dict_inputs_matching_dicts(self):
        """Test forward with matching dictionary inputs"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }
        targets = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }

        loss = loss_wrapper.forward(outputs, targets)

        # Perfect match, loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_forward_dict_inputs_with_nans(self):
        """Test forward with dictionary inputs containing NaN values"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }
        targets = {
            "class1": torch.tensor([[1.0, float("nan")]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }

        loss = loss_wrapper.forward(outputs, targets)

        # Loss from class1: only first element counted (0)
        # Loss from class2: both elements (0)
        # Average of 0 and 0 = 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_forward_dict_targets_list_outputs(self):
        """Test forward with dict targets and list/tuple outputs"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
        targets = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }

        loss = loss_wrapper.forward(outputs, targets)

        # Perfect match, loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_bce_loss_with_nans(self):
        """Test with BCE loss and NaN values"""
        loss_wrapper = CellMapLossWrapper(torch.nn.BCELoss)
        outputs = torch.tensor([[0.5, 0.8], [0.3, 0.9]])
        targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])

        loss = loss_wrapper.calc_loss(outputs, targets)

        # Loss should be computed only for non-NaN values
        assert not torch.isnan(loss)
        assert loss >= 0  # BCE loss is always non-negative
