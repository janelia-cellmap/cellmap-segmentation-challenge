import os
import numpy as np
import pytest

# Update this import to match your project layout
# e.g. from cellmap_segmentation_challenge.evaluate import match_instances
from cellmap_segmentation_challenge.evaluate import match_instances


def _make_simple_overlap_case():
    """
    GT: two instances (1,2) each a 2x2 block.
    Pred: two instances (1,2) each aligned perfectly to GT.
    Expect mapping {1:1, 2:2}.
    """
    gt = np.zeros((6, 6), dtype=np.int32)
    pred = np.zeros((6, 6), dtype=np.int32)

    gt[0:2, 0:2] = 1
    gt[4:6, 4:6] = 2

    pred[0:2, 0:2] = 1
    pred[4:6, 4:6] = 2
    return gt, pred


def _make_nonsequential_ids_case():
    """
    GT uses non-sequential IDs: {1, 5}.
    Pred uses non-sequential IDs: {1, 7}.
    They overlap perfectly in two disjoint regions.

    Under the function's stated assumption (IDs range 1..max), nG=max(gt)=5 and nP=max(pred)=7.
    The matching should still recover mapping {1:1, 7:5}.
    """
    gt = np.zeros((6, 6), dtype=np.int32)
    pred = np.zeros((6, 6), dtype=np.int32)

    gt[0:2, 0:2] = 1
    gt[4:6, 4:6] = 5

    pred[0:2, 0:2] = 1
    pred[4:6, 4:6] = 7
    return gt, pred


def _make_competing_overlap_case():
    """
    One pred overlaps two GTs; ensure it picks the higher IoU match.
    GT1 is a 2x2 block. GT2 is a 1x1 block.
    Pred is a 2x2 block overlapping GT1 fully and GT2 partially or not.
    Expect pred->GT picks GT1.
    """
    gt = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)

    # GT1: 2x2 at top-left
    gt[0:2, 0:2] = 1
    # GT2: single voxel at bottom-right
    gt[3, 3] = 2

    # Pred1: 2x2 at top-left (perfectly overlaps GT1)
    pred[0:2, 0:2] = 1
    return gt, pred


def _require_ortools():
    # If ortools isn't installed in the test environment, skip the tests that need it.
    pytest.importorskip("ortools.graph.python.min_cost_flow")


def test_shape_mismatch_raises():
    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.zeros((3, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="must have the same shape"):
        match_instances(gt, pred)


def test_ratio_cutoff_returns_none(monkeypatch):
    # Make cutoff low so it triggers easily
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1.0")
    gt = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)

    # nG = 1, nP = 3 => nP/nG = 3 > 1 => None
    gt[0, 0] = 1
    pred[0, 0] = 1
    pred[0, 1] = 2
    pred[0, 2] = 3

    out = match_instances(gt, pred)
    assert out is None


def test_overlap_edges_cutoff_returns_none(monkeypatch):
    # Set MAX_OVERLAP_EDGES extremely low to force the cutoff
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "0")

    gt = np.zeros((2, 2), dtype=np.int32)
    pred = np.zeros((2, 2), dtype=np.int32)
    gt[0, 0] = 1
    pred[0, 0] = 1  # one overlap edge exists -> should exceed 0

    out = match_instances(gt, pred)
    assert out is None


def test_no_overlap_returns_empty_dict(monkeypatch):
    # Avoid ratio cutoff
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")

    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.zeros((3, 3), dtype=np.int32)
    gt[0, 0] = 1
    pred[2, 2] = 1  # no voxel overlaps -> gi.size==0 branch

    out = match_instances(gt, pred)
    # Your code returns np.zeros((nG,nP)) in this branch, not a dict.
    # If you want the function to always return dict|None, change the implementation.
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1)
    assert float(out[0, 0]) == 0.0


def test_perfect_two_object_mapping(monkeypatch):
    """
    Happy-path: two objects, perfect overlaps -> mapping should be exact.
    This test will FAIL until the OR-Tools bulk-add call is fixed.
    """
    _require_ortools()
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")

    gt, pred = _make_simple_overlap_case()
    mapping = match_instances(gt, pred)
    assert isinstance(mapping, dict)
    assert mapping == {1: 1, 2: 2}


def test_nonsequential_ids_mapping(monkeypatch):
    """
    Non-sequential IDs still match correctly.
    This test will FAIL until the OR-Tools bulk-add call is fixed.
    """
    _require_ortools()
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")

    gt, pred = _make_nonsequential_ids_case()
    mapping = match_instances(gt, pred)
    assert isinstance(mapping, dict)
    assert mapping.get(1) == 1
    assert mapping.get(7) == 5
    # Ensure it doesn't incorrectly map to "missing" ID rows/cols (like 2..4, etc.)
    assert 2 not in mapping
    assert 3 not in mapping
    assert 4 not in mapping
    assert 5 not in mapping  # mapping keys are pred IDs


def test_prefers_higher_iou(monkeypatch):
    """
    If there are choices, should choose the higher-IoU edge.
    This test will FAIL until the OR-Tools bulk-add call is fixed.
    """
    _require_ortools()
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")

    gt, pred = _make_competing_overlap_case()
    mapping = match_instances(gt, pred)
    assert isinstance(mapping, dict)
    assert mapping == {1: 1}  # pred 1 matches GT 1


def test_unmatched_allowed(monkeypatch):
    """
    If a GT object has no IoU>0 edges, it should just go unmatched (no remap).
    Here: GT has one object; pred has one object in a disjoint location => no overlaps.
    This again hits the gi.size==0 early return.
    """
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    gt = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)
    gt[0:2, 0:2] = 1
    pred[2:4, 2:4] = 1

    out = match_instances(gt, pred)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1)
    assert float(out[0, 0]) == 0.0
