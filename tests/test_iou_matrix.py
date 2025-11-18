import numpy as np
import pytest

# Change this import to wherever your implementation lives
from cellmap_segmentation_challenge.evaluate import iou_matrix

INSTANCE_RATIO_CUTOFF = float(os.getenv("INSTANCE_RATIO_CUTOFF", 50))


def test_basic_iou_small_grid(monkeypatch):
    """
    gt IDs: 1..2 ; pred IDs: 1..2 ; 0 is background
    Expected IoU:
      [[0.5, 0.0],
       [0.0, 0.75]]
    """
    gt = np.array(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    pred = np.array(
        [
            [0, 1, 0],
            [0, 2, 2],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    # Make the cutoff permissive so it doesn't trip in this test
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "999")

    iou = iou_matrix(gt, pred)
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(
        iou, np.array([[0.5, 0.0], [0.0, 0.75]], dtype=np.float32), rtol=1e-6, atol=1e-6
    )
    assert iou.dtype == np.float32


def test_empty_pred():
    gt = np.array(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    pred = np.zeros_like(gt)
    iou = iou_matrix(gt, pred)
    # nG=2, nP=0 → (2,0) array of zeros
    assert iou.shape == (2, 0)
    assert iou.size == 0


def test_empty_gt():
    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.array(
        [
            [0, 1, 0],
            [0, 2, 2],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    iou = iou_matrix(gt, pred)
    # nG=0, nP=2 → (0,2) array of zeros
    assert iou.shape == (0, 2)
    assert iou.size == 0


def test_rectangle_more_preds(monkeypatch):
    """
    Non-square case: nG=2, nP=3; verify shape and a couple of values.
    """
    gt = np.array(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    pred = np.array(
        [
            [0, 1, 3],
            [0, 2, 3],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    # Ensure cutoff won't trigger
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "999")
    iou = iou_matrix(gt, pred)
    assert iou.shape == (2, 3)

    # Quick sanity checks:
    # gt1∩pred1 = 1 pixel, |gt1|=2, |pred1|=1 → IoU = 1/(2+1-1)=0.5
    # gt2∩pred2 >= 2, |gt2|=3, |pred2|>=2 → IoU should be > 0
    assert abs(float(iou[0, 0]) - 0.5) < 1e-6
    assert iou[1, 1] > 0.0


def test_ratio_cutoff_triggers(monkeypatch):
    """
    Force the INSTANCE_RATIO_CUTOFF branch to return None.
    """
    gt = np.array(
        [
            [1, 1, 0],
            [2, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )  # nG = 2
    pred = np.array(
        [
            [1, 2, 3],
            [4, 5, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )  # nP = 5

    # Set cutoff small so nP/nG = 2.5 exceeds it
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "0.1")

    out = iou_matrix(gt, pred)
    assert out is None


def test_matches_naive_reference():
    """
    Cross-check against a naive python/dict reference on a tiny random mask.
    """
    rng = np.random.default_rng(0)
    H, W = 6, 7
    nG, nP = 3, 4
    gt = rng.integers(0, nG + 1, size=(H, W), dtype=np.int32)
    pred = rng.integers(0, nP + 1, size=(H, W), dtype=np.int32)

    # Fast impl
    fast = iou_matrix(gt, pred)

    # Naive reference
    gt_counts = {k: int((gt == k).sum()) for k in range(1, nG + 1)}
    pr_counts = {k: int((pred == k).sum()) for k in range(1, nP + 1)}
    inter = np.zeros((nG, nP), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            g = int(gt[y, x])
            p = int(pred[y, x])
            if g > 0 and p > 0:
                inter[g - 1, p - 1] += 1
    gt_sizes = np.array([gt_counts[i] for i in range(1, nG + 1)])[:, None]
    pr_sizes = np.array([pr_counts[j] for j in range(1, nP + 1)])[None, :]
    union = gt_sizes + pr_sizes - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = np.where(union > 0, inter / union, 0.0).astype(np.float32)

    np.testing.assert_allclose(fast, ref, rtol=1e-6, atol=1e-6)
