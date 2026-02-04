from dataclasses import dataclass
from pathlib import Path

import cc3d
import numpy as np
import zarr
from fastremap import unique
import pytest
from concurrent.futures import Future


class DummySerialExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            result = fn(*args, **kwargs)
            f.set_result(result)
        except Exception as e:
            f.set_exception(e)
        return f

    def shutdown(self, wait=True):
        pass


@pytest.fixture(autouse=True)
def patch_executor(monkeypatch):
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummySerialExecutor)


# ------------------------
# Helpers / tiny dataclasses
# ------------------------
@dataclass
class DummyCrop:
    voxel_size: tuple
    shape: tuple
    translation: tuple


def _hausdorff_full_reference_labels(
    truth_label: np.ndarray,
    pred_label: np.ndarray,
    tid: int,
    voxel_size,
    max_distance: float,
    method: str = "standard",
    percentile: float | None = None,
) -> float:
    """
    Full-volume reference for a single tid, matching the ROI version semantics.
    """
    from cellmap_segmentation_challenge import evaluate as ev
    from scipy.ndimage import distance_transform_edt

    a = truth_label == tid
    b = pred_label == tid

    a_n = int(a.sum())
    b_n = int(b.sum())
    if a_n == 0 and b_n == 0:
        return 0.0
    if a_n == 0 or b_n == 0:
        return float(max_distance)

    ndim = truth_label.ndim
    vs = np.asarray(voxel_size, dtype=np.float64)
    if vs.size != ndim:
        vs = vs[-ndim:]

    dist_to_b = distance_transform_edt(~b, sampling=vs)
    dist_to_a = distance_transform_edt(~a, sampling=vs)

    fwd = dist_to_b[a]
    bwd = dist_to_a[b]

    if method == "standard":
        d = max(fwd.max(initial=0.0), bwd.max(initial=0.0))
    elif method == "modified":
        d = max(fwd.mean() if fwd.size else 0.0, bwd.mean() if bwd.size else 0.0)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("'percentile' must be provided when method='percentile'")
        d = max(
            float(np.percentile(fwd, percentile)) if fwd.size else 0.0,
            float(np.percentile(bwd, percentile)) if bwd.size else 0.0,
        )
    else:
        raise ValueError("method must be one of {'standard', 'modified', 'percentile'}")

    return float(min(d, max_distance))


# ------------------------
# ROI Hausdorff tests
# ------------------------


def test_roi_hausdorff_identical_instance_is_zero():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.zeros((5, 5), dtype=np.int32)
    pred = np.zeros((5, 5), dtype=np.int32)
    tid = 1

    truth[2, 2] = tid
    pred[2, 2] = tid

    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)

    d = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=(1.0, 1.0),
        max_distance=10.0,
        method="standard",
    )
    assert np.isclose(d, 0.0)


def test_roi_hausdorff_matches_full_reference_standard_and_modified():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.zeros((1, 20), dtype=np.int32)
    pred = np.zeros((1, 20), dtype=np.int32)
    tid = 7

    truth[0, 1] = tid
    pred[0, 4] = tid  # distance 3

    voxel_size = (1.0, 1.0)
    max_distance = 100.0

    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)

    d_roi = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="standard",
    )
    d_ref = _hausdorff_full_reference_labels(
        truth,
        pred,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="standard",
    )
    assert np.isclose(d_roi, d_ref)
    assert np.isclose(d_roi, 3.0)

    d_roi_mod = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="modified",
    )
    d_ref_mod = _hausdorff_full_reference_labels(
        truth,
        pred,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="modified",
    )
    assert np.isclose(d_roi_mod, d_ref_mod)
    assert np.isclose(d_roi_mod, 3.0)


def test_roi_hausdorff_percentile_matches_full_reference():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.zeros((1, 8), dtype=np.int32)
    pred = np.zeros((1, 8), dtype=np.int32)
    tid = 2

    # truth tid at 0,1; pred tid at 4,5
    truth[0, 0] = tid
    truth[0, 1] = tid
    pred[0, 4] = tid
    pred[0, 5] = tid

    voxel_size = (1.0, 1.0)
    max_distance = 100.0

    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)

    d_roi = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="percentile",
        percentile=50,
    )
    d_ref = _hausdorff_full_reference_labels(
        truth,
        pred,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="percentile",
        percentile=50,
    )
    assert np.isclose(d_roi, d_ref, atol=1e-6)


def test_roi_hausdorff_empty_sets_and_missing_instance():
    from cellmap_segmentation_challenge import evaluate as ev

    voxel_size = (1.0, 1.0)
    max_distance = 5.0
    tid = 3

    truth = np.zeros((6, 6), dtype=np.int32)
    pred = np.zeros_like(truth)

    # present only in truth -> infinity
    truth[0, 0] = tid
    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)
    d1 = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
    )
    assert np.isclose(d1, max_distance)


def test_roi_hausdorff_clips_to_max_distance_matches_reference():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.zeros((1, 30), dtype=np.int32)
    pred = np.zeros((1, 30), dtype=np.int32)
    tid = 1

    truth[0, 0] = tid
    pred[0, 10] = tid  # true distance 10

    voxel_size = (1.0, 1.0)
    max_distance = 3.0

    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)

    d_roi = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="standard",
    )
    assert np.isclose(d_roi, max_distance)

    d_ref = _hausdorff_full_reference_labels(
        truth,
        pred,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="standard",
    )
    assert np.isclose(d_roi, d_ref)


def test_roi_hausdorff_anisotropic_voxel_size_matches_reference():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.zeros((5, 5), dtype=np.int32)
    pred = np.zeros((5, 5), dtype=np.int32)
    tid = 9

    truth[1, 1] = tid
    pred[1, 3] = tid  # 2 steps along x

    voxel_size = (2.0, 0.5)  # physical distance = 2 * 0.5 = 1.0
    max_distance = 100.0

    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)

    d_roi = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="standard",
    )
    d_ref = _hausdorff_full_reference_labels(
        truth,
        pred,
        tid,
        voxel_size=voxel_size,
        max_distance=max_distance,
        method="standard",
    )
    assert np.isclose(d_roi, d_ref, atol=1e-6)
    assert np.isclose(d_roi, 1.0, atol=1e-6)


@pytest.mark.usefixtures("monkeypatch")
def test_roi_none_returns_inf(monkeypatch):
    """
    Forces roi_slices_for_pair to return None to exercise that branch.
    Should return infinity when the instance exists in truth or pred
    (i.e. not the "both absent" case).
    """
    from cellmap_segmentation_challenge import evaluate as ev

    def _fake_roi(*args, **kwargs):
        return None

    monkeypatch.setattr(ev, "roi_slices_for_pair", _fake_roi)

    truth = np.zeros((5, 5), dtype=np.int32)
    pred = np.zeros_like(truth)
    tid = 1

    # Make tid present in exactly one volume so we don't trigger the "both absent -> 0" shortcut
    truth[2, 2] = tid  # present in truth only

    truth_stats = cc3d.statistics(truth)
    pred_stats = cc3d.statistics(pred)
    max_distance = 7.0

    d = ev.compute_hausdorff_distance_roi(
        truth,
        truth_stats,
        pred,
        pred_stats,
        tid,
        voxel_size=(1.0, 1.0),
        max_distance=max_distance,
    )
    assert np.isclose(d, max_distance)


def test_optimized_hausdorff_distances_per_instance():
    from cellmap_segmentation_challenge import evaluate as ev

    # two instances, perfectly matched
    truth = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    pred = truth.copy()
    voxel_size = (1.0, 1.0)

    dists = ev.optimized_hausdorff_distances(
        truth, pred, voxel_size, hausdorff_distance_max=np.inf, method="standard"
    )
    # there are two non-zero ids
    ids = unique(truth)
    ids = ids[ids != 0]
    assert dists.shape == (ids.size,)
    assert np.allclose(dists, 0.0)


# ------------------------
# score_instance tests
# ------------------------


def test_score_instance_perfect_match():
    from cellmap_segmentation_challenge import evaluate as ev

    label = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    scores = ev.score_instance(label, label, voxel_size=(1.0, 1.0))

    assert np.isclose(scores["mean_accuracy"], 1.0)
    assert np.isclose(scores["hausdorff_distance"], 0.0)
    assert np.isclose(scores["normalized_hausdorff_distance"], 1.0)
    assert np.isclose(scores["combined_score"], 1.0)


def test_score_instance_simple_shift():
    from cellmap_segmentation_challenge import evaluate as ev

    # GT has one instance [0,0] and [0,1]
    truth = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int32)
    # Prediction shifted one voxel to the right: [0,1],[0,2] (and needs renumbering)
    pred = np.array([[0, 2, 2], [0, 0, 0]], dtype=np.int32)
    voxel_size = (1.0, 1.0)
    scores = ev.score_instance(pred, truth, voxel_size)

    # Accuracy should not be 1 but positive
    assert 0.0 < scores["mean_accuracy"] < 1.0
    # Hausdorff distance is 1 (each point moves 1 voxel)
    assert np.isclose(scores["hausdorff_distance"], 1.0)


# ------------------------
# score_semantic tests
# ------------------------


def test_score_semantic_perfect_match():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.array([[0, 1], [1, 1]], dtype=float)
    pred = truth.copy()
    scores = ev.score_semantic(pred, truth)
    assert np.isclose(scores["iou"], 1.0)
    assert np.isclose(scores["dice_score"], 1.0)


def test_score_semantic_partial_overlap():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.array([[0, 1], [1, 1]], dtype=float)
    # prediction misses one positive voxel
    pred = np.array([[0, 1], [0, 1]], dtype=float)

    scores = ev.score_semantic(pred, truth)

    # manual IoU: TP = 2, FP = 0, FN = 1 -> IoU = 2 / (2+0+1) = 2/3
    assert np.isclose(scores["iou"], 2 / 3)
    # manual Dice: 2TP / (2TP + FP + FN) = 4 / (4 + 0 + 1) = 0.8
    assert np.isclose(scores["dice_score"], 0.8)


def test_score_semantic_no_foreground():
    from cellmap_segmentation_challenge import evaluate as ev

    truth = np.zeros((3, 3), dtype=float)
    pred = np.zeros_like(truth)
    scores = ev.score_semantic(pred, truth)
    assert np.isclose(scores["iou"], 1.0)
    assert np.isclose(scores["dice_score"], 1.0)


# ------------------------
# resize_array tests
# ------------------------


def test_resize_array_pad_and_crop():
    from cellmap_segmentation_challenge import evaluate as ev

    arr = np.ones((2, 2), dtype=np.int32)
    # First confirm padding up to (4,4)
    padded = ev.resize_array(arr, (4, 4), pad_value=0)
    assert padded.shape == (4, 4)
    # original ones centered
    assert np.all(padded[1:3, 1:3] == 1)

    # Crop down from (4,4) to (2,2) (should give central 2x2)
    cropped = ev.resize_array(padded, (2, 2), pad_value=0)
    assert cropped.shape == (2, 2)
    assert np.all(cropped == 1)


def test_resize_array_only_crop():
    from cellmap_segmentation_challenge import evaluate as ev

    arr = np.arange(16).reshape(4, 4)
    target = (2, 2)
    out = ev.resize_array(arr, target)

    assert out.shape == target
    # Center crop: indices [1:3, 1:3]
    expected = arr[1:3, 1:3]
    assert np.array_equal(out, expected)


# ------------------------
# match_crop_space tests (single-scale)
# ------------------------


def test_match_crop_space_no_rescale_no_translation(tmp_path):
    """Single-scale array with matching voxel size and translation=0: should just crop/pad."""
    from cellmap_segmentation_challenge import evaluate as ev

    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    root = tmp_path / "vol.zarr"
    ds = zarr.open(str(root), mode="w", shape=arr.shape, dtype=arr.dtype)
    ds[:] = arr
    ds.attrs["voxel_size"] = (1.0, 1.0)
    ds.attrs["translation"] = (0.0, 0.0)

    # same voxel size, same shape
    out = ev.match_crop_space(
        str(root),
        "sem_label",
        voxel_size=(1.0, 1.0),
        shape=(4, 4),
        translation=(0.0, 0.0),
    )
    assert np.array_equal(out, arr)

    # same voxel size, smaller shape -> centered crop
    out2 = ev.match_crop_space(
        str(root),
        "sem_label",
        voxel_size=(1.0, 1.0),
        shape=(2, 2),
        translation=(1.0, 1.0),
    )
    assert out2.shape == (2, 2)
    assert np.array_equal(out2, arr[1:3, 1:3])


def test_match_crop_space_rescale_instance(tmp_path):
    """
    Simple rescale case for instance label:
    input voxel_size=(2,2), target=(1,1). We expect a 2x upsample along each axis.
    """
    from cellmap_segmentation_challenge import evaluate as ev

    arr = np.zeros((2, 2), dtype=np.uint8)
    arr[0, 0] = 1
    root = tmp_path / "inst.zarr"
    ds = zarr.open(str(root), mode="w", shape=arr.shape, dtype=arr.dtype)
    ds[:] = arr
    ds.attrs["voxel_size"] = (2.0, 2.0)
    ds.attrs["translation"] = (0.0, 0.0)

    out_shape = (4, 4)
    out = ev.match_crop_space(
        str(root),
        "instance",
        voxel_size=(1.0, 1.0),
        shape=out_shape,
        translation=(0.0, 0.0),
    )
    assert out.shape == out_shape
    # nearest neighbor upsampling should preserve instance id in top-left 2x2 region
    assert np.all(out[0:2, 0:2] == 1)


# ------------------------
# empty_label_score & missing_volume_score
# ------------------------


def _create_simple_volume(
    path: Path,
    crop_name: str,
    label_name: str,
    arr: np.ndarray,
    voxel_size=(1.0, 1.0, 1.0),
):
    ds = zarr.open(
        str(path / crop_name),
        path=label_name,
        mode="w",
        shape=arr.shape,
        dtype=arr.dtype,
    )
    ds[:] = arr
    ds.attrs["voxel_size"] = voxel_size


def test_empty_label_score_instance(tmp_path):
    from cellmap_segmentation_challenge import evaluate as ev

    truth_root = tmp_path / "truth.zarr"
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    _create_simple_volume(truth_root, "crop1", "instance", arr)

    scores = ev.empty_label_score(
        label="instance",
        crop_name="crop1",
        instance_classes=["instance"],
        truth_path=truth_root.as_posix(),
    )
    # num_voxels should match volume size
    assert scores["num_voxels"] == arr.size
    assert scores["is_missing"] is True
    assert scores["mean_accuracy"] == 0


def test_missing_volume_score_mixed_labels(tmp_path):
    from cellmap_segmentation_challenge import evaluate as ev

    truth_root = tmp_path / "truth_volume.zarr"
    arr_inst = np.zeros((2, 2, 2), dtype=np.uint8)
    arr_sem = np.zeros((2, 2, 2), dtype=np.uint8)

    _create_simple_volume(truth_root, "crop1", "instance", arr_inst)
    _create_simple_volume(truth_root, "crop1", "sem", arr_sem)

    scores = ev.missing_volume_score(
        truth_path=truth_root,
        volume="crop1",
        instance_classes=["instance"],
    )

    assert set(scores.keys()) == {"instance", "sem"}
    assert scores["instance"]["is_missing"] is True
    assert scores["sem"]["is_missing"] is True
    assert scores["instance"]["mean_accuracy"] == 0.0
    assert scores["sem"]["iou"] == 0.0


# ------------------------
# combine_scores tests
# ------------------------


def test_combine_scores_instance_and_semantic():
    from cellmap_segmentation_challenge import evaluate as ev

    # Two volumes, one instance, one semantic
    scores = {
        "crop1": {
            "instance": {
                "mean_accuracy": 1.0,
                "hausdorff_distance": 0.0,
                "normalized_hausdorff_distance": 1.0,
                "combined_score": 1.0,
                "num_voxels": 8,
                "voxel_size": (1.0, 1.0, 1.0),
                "is_missing": False,
            }
        },
        "crop2": {
            "sem": {
                "iou": 0.5,
                "dice_score": 2 / 3,
                "num_voxels": 8,
                "voxel_size": (1.0, 1.0, 1.0),
                "is_missing": False,
            }
        },
    }

    combined = ev.combine_scores(
        scores, include_missing=True, instance_classes=["instance"]
    )

    ls = combined["label_scores"]
    assert np.isclose(ls["instance"]["combined_score"], 1.0)
    assert np.isclose(ls["sem"]["iou"], 0.5)
    assert "overall_instance_score" in combined
    assert "overall_semantic_score" in combined
    assert "overall_score" in combined
    # only one of each type -> overall scores are just these
    assert np.isclose(combined["overall_instance_score"], 1.0)
    assert np.isclose(combined["overall_semantic_score"], 0.5)


# ------------------------
# score_label & score_submission-style integration
# ------------------------


@pytest.mark.usefixtures("monkeypatch")
def test_score_label_instance_integration(monkeypatch, tmp_path):
    """
    Small integration test: zarr truth + pred, dummy TEST_CROPS_DICT entry,
    score_label should return instance metrics consistent with score_instance.
    """
    from cellmap_segmentation_challenge import evaluate as ev

    # Arrange mini truth volume
    crop_name = "crop1"
    label_name = "instance"
    truth_root = tmp_path / "truth.zarr"

    arr = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.uint8)
    _create_simple_volume(truth_root, crop_name, label_name, arr)

    # matching pred volume with same data
    pred_root = tmp_path / "pred.zarr"
    _create_simple_volume(pred_root, crop_name, label_name, arr)

    # monkeypatch TEST_CROPS_DICT for this one crop/label
    dummy_crop = DummyCrop(
        voxel_size=(1.0, 1.0, 1.0), shape=arr.shape, translation=(0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(
        ev,
        "TEST_CROPS_DICT",
        {(1, label_name): dummy_crop},
        raising=False,
    )

    crop_name_str = crop_name  # "crop1"
    pred_label_path = ev.UPath(pred_root.as_posix()) / crop_name_str / label_name

    crop_out, label_out, results = ev.score_label(
        pred_label_path=pred_label_path,
        label_name=label_name,
        crop_name=crop_name_str,
        truth_path=ev.UPath(truth_root.as_posix()),
        instance_classes=["instance"],
    )

    assert crop_out == crop_name_str
    assert label_out == label_name
    assert np.isclose(results["mean_accuracy"], 1.0)
    assert np.isclose(results["hausdorff_distance"], 0.0)
    assert results["is_missing"] is False


@pytest.mark.usefixtures("monkeypatch")
def test_score_submission(monkeypatch, tmp_path):
    """
    End-to-end-ish test:
      - create a truth volume with a single crop and label
      - create matching prediction volume
      - zip it
    """
    from cellmap_segmentation_challenge import evaluate as ev
    from cellmap_segmentation_challenge.utils import zip_submission

    # Create truth.zarr/crop1/instance
    truth_root = tmp_path / "truth.zarr"
    crop_name = "crop1"
    label_name = "instance"

    arr = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.uint8)
    _create_simple_volume(truth_root, crop_name, label_name, arr)

    # Create submission zarr directory structure: submission_root/crop1/instance
    submission_root = tmp_path / "submission.zarr"
    _create_simple_volume(submission_root, crop_name, label_name, arr)

    # Patch TEST_CROPS_DICT
    dummy_crop = DummyCrop(
        voxel_size=(1.0, 1.0, 1.0), shape=arr.shape, translation=(0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(
        ev,
        "TEST_CROPS_DICT",
        {(1, label_name): dummy_crop},
        raising=False,
    )

    assert (
        1,
        label_name,
    ) in ev.TEST_CROPS_DICT, f"Key (1, {label_name}) is missing in TEST_CROPS_DICT"

    # Zip the submission_root contents so that unzip_file will create
    # a directory with crop1 directly inside.
    zip_path = zip_submission(submission_root)

    # Run score_submission with explicit truth_path and instance_classes
    scores = ev.score_submission(
        submission_path=zip_path.as_posix(),
        result_file=None,
        truth_path=truth_root.as_posix(),
        instance_classes=[label_name],
    )

    # We expect perfect instance score
    assert np.isclose(scores["overall_instance_score"], 1.0)
    # No semantic labels -> overall_semantic_score is nan, but that’s fine;
    # just ensure label_scores present and correct.
    assert "label_scores" in scores
    assert np.isclose(scores["label_scores"][label_name]["mean_accuracy"], 1.0)


@pytest.mark.usefixtures("monkeypatch")
def test_score_submission_json_output(monkeypatch, tmp_path):
    """
    Test that score_submission results can be written to a JSON file and read back successfully.
    """
    import json
    from cellmap_segmentation_challenge import evaluate as ev
    from cellmap_segmentation_challenge.utils import zip_submission

    # Create truth.zarr/crop1/instance
    truth_root = tmp_path / "truth.zarr"
    crop_name = "crop1"
    label_name = "instance"
    arr = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.uint8)
    _create_simple_volume(truth_root, crop_name, label_name, arr)

    # Create submission zarr directory structure: submission_root/crop1/instance
    submission_root = tmp_path / "submission.zarr"
    _create_simple_volume(submission_root, crop_name, label_name, arr)

    # Patch TEST_CROPS_DICT
    dummy_crop = DummyCrop(
        voxel_size=(1.0, 1.0, 1.0), shape=arr.shape, translation=(0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(
        ev,
        "TEST_CROPS_DICT",
        {(1, label_name): dummy_crop},
        raising=False,
    )

    assert (
        1,
        label_name,
    ) in ev.TEST_CROPS_DICT, f"Key (1, {label_name}) is missing in TEST_CROPS_DICT"

    # Zip the submission_root contents
    zip_path = zip_submission(submission_root)

    # Write results to a file
    result_file = tmp_path / "results.json"
    ev.score_submission(
        submission_path=zip_path.as_posix(),
        result_file=result_file.as_posix(),
        truth_path=truth_root.as_posix(),
        instance_classes=[label_name],
    )

    # Check that the file exists and contains valid JSON
    assert result_file.exists()
    with open(result_file, "r") as f:
        loaded = json.load(f)
    assert isinstance(loaded, dict)
    assert "label_scores" in loaded
