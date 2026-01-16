# tests/test_match_instance.py
import importlib

import numpy as np
import pytest

MODULE = "cellmap_segmentation_challenge.evaluate"


def _reload_module():
    """
    Reload evaluate.py so env-var cutoffs are re-read even if the module caches them at import time.
    """
    mod = importlib.import_module(MODULE)
    return importlib.reload(mod)


def _require_ortools():
    pytest.importorskip("ortools.graph.python.min_cost_flow")


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
    Expect mapping {1:1, 7:5}.
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

    GT1: 2x2 block at top-left.
    GT2: single voxel at (1,1) (inside GT1)?? No, we need disjoint GTs.
    Instead: two GTs that both overlap pred but with different IoU.

    Setup:
      - GT1: 2x2 at top-left (4 px)
      - GT2: 2x1 at top-right (2 px)
      - Pred1: 2x3 across top row block overlapping both:
              overlaps GT1 by 4 and GT2 by 2, but IoU differs due to unions.
    Expect Pred1 chooses GT1.
    """
    gt = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)

    gt[0:2, 0:2] = 1  # 4 px
    gt[0:2, 2:3] = 2  # 2 px (a 2x1 column)

    pred[0:2, 0:3] = 1  # 6 px overlaps GT1 (4) and GT2 (2)
    return gt, pred


def test_shape_mismatch_raises():
    ev = _reload_module()
    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.zeros((3, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="must have the same shape"):
        ev.match_instances(gt, pred)


def test_ratio_cutoff_returns_none(monkeypatch):
    """
    If INSTANCE_RATIO_CUTOFF is exceeded, function returns None.
    We reload the module after setting env to avoid import-time caching issues.
    """
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1.0")
    monkeypatch.setenv("INITIAL_INSTANCE_RATIO_CUTOFF", "1.0")
    monkeypatch.setenv("FINAL_INSTANCE_RATIO_CUTOFF", "1.0")
    monkeypatch.setenv("INSTANCE_RATIO_FACTOR", "1.0")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")
    ev = _reload_module()

    gt = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)

    # nG = 1, nP = 3 => ratio=3 > 1 => None
    gt[0, 0] = 1
    pred[0, 0] = 1
    pred[0, 1] = 2
    pred[0, 2] = 3

    out = ev.match_instances(gt, pred)
    assert out is None


def test_overlap_edges_cutoff_returns_none(monkeypatch):
    """
    If MAX_OVERLAP_EDGES is exceeded, function returns None.
    """
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "0")
    ev = _reload_module()

    gt = np.zeros((2, 2), dtype=np.int32)
    pred = np.zeros((2, 2), dtype=np.int32)
    gt[0, 0] = 1
    pred[0, 0] = 1  # one overlap edge exists -> should exceed 0

    out = ev.match_instances(gt, pred)
    assert out is None


def test_no_overlap_returns_empty_iou_matrix(monkeypatch):
    """
    When there are GT and Pred instances but no fg-fg overlap (gi.size==0),
    current implementation returns a dense np.zeros((nG, nP)).
    """
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")
    ev = _reload_module()

    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.zeros((3, 3), dtype=np.int32)
    gt[0, 0] = 1
    pred[2, 2] = 1  # disjoint => no overlap

    out = ev.match_instances(gt, pred)
    assert isinstance(out, dict)
    assert out == {}


def test_unmatched_allowed_hits_no_overlap_branch(monkeypatch):
    """
    GT has one object; pred has one object disjoint -> no overlap branch.
    """
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")
    ev = _reload_module()

    gt = np.zeros((4, 4), dtype=np.int32)
    pred = np.zeros((4, 4), dtype=np.int32)
    gt[0:2, 0:2] = 1
    pred[2:4, 2:4] = 1

    out = ev.match_instances(gt, pred)
    assert isinstance(out, dict)
    assert out == {}


# -------------------------
# OR-Tools / matching tests
# -------------------------


def test_perfect_two_object_mapping(monkeypatch):
    _require_ortools()
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")
    ev = _reload_module()

    gt, pred = _make_simple_overlap_case()
    mapping = ev.match_instances(gt, pred)

    assert isinstance(mapping, dict)
    assert mapping == {1: 1, 2: 2}


def test_nonsequential_ids_mapping(monkeypatch):
    _require_ortools()
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")
    ev = _reload_module()

    gt, pred = _make_nonsequential_ids_case()
    mapping = ev.match_instances(gt, pred)

    assert isinstance(mapping, dict)
    assert mapping == {1: 1, 7: 5}

    # Ensure it doesn't create mappings for missing pred IDs (2..6)
    for k in [2, 3, 4, 5, 6]:
        assert k not in mapping


def test_prefers_higher_iou(monkeypatch):
    _require_ortools()
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1000")
    monkeypatch.setenv("MAX_OVERLAP_EDGES", "5000000")
    ev = _reload_module()

    gt, pred = _make_competing_overlap_case()
    mapping = ev.match_instances(gt, pred)

    assert isinstance(mapping, dict)
    # Only pred id 1 exists; it should match GT 1 (higher IoU than GT 2).
    assert mapping == {1: 1}
