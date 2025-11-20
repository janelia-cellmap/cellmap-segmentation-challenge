import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from fastremap import unique

from cellmap_segmentation_challenge import evaluate as ev
from cellmap_segmentation_challenge.utils import zip_submission


# ------------------------
# Helpers / tiny dataclasses
# ------------------------


@dataclass
class DummyCrop:
    voxel_size: tuple
    shape: tuple
    translation: tuple


# ------------------------
# iou_matrix tests
# ------------------------


def test_iou_matrix_basic():
    """
    gt:
      0 1
      2 2

    pred:
      0 1
      0 2

    gt id 1: 1 voxel, pred id 1: 1 voxel, intersection 1 -> IoU 1
    gt id 2: 2 voxels, pred id 2: 1 voxel, intersection 1 -> IoU 1/2
    """
    gt = np.array([[0, 1], [2, 2]], dtype=np.int32)
    pred = np.array([[0, 1], [0, 2]], dtype=np.int32)

    iou = ev.iou_matrix(gt, pred)

    assert iou.shape == (2, 2)
    # (gt1, pred1)
    assert np.isclose(iou[0, 0], 1.0)
    # (gt2, pred2) = 1 / (2 + 1 - 1) = 0.5
    assert np.isclose(iou[1, 1], 0.5)
    # all other entries should be 0
    assert np.isclose(iou[0, 1], 0.0)
    assert np.isclose(iou[1, 0], 0.0)


def test_iou_matrix_no_gt_instances():
    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.array([[0, 1, 1], [0, 0, 2], [0, 0, 0]], dtype=np.int32)

    iou = ev.iou_matrix(gt, pred)
    # nG = 0, nP = 2 -> shape (0,2)
    assert iou.shape == (0, 2)


def test_iou_matrix_no_pred_instances():
    gt = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
    pred = np.zeros_like(gt)

    iou = ev.iou_matrix(gt, pred)
    # nG = 2, nP = 0 -> shape (2,0)
    assert iou.shape == (2, 0)


def test_iou_matrix_too_many_pred_instances(monkeypatch):
    # force INSTANCE_RATIO_CUTOFF low to trigger None
    monkeypatch.setenv("INSTANCE_RATIO_CUTOFF", "1")
    gt = np.array([[0, 1, 0, 0]], dtype=np.int32)  # 1 instance
    pred = np.array([[0, 1, 2, 3]], dtype=np.int32)  # 3 instances

    res = ev.iou_matrix(gt, pred)
    assert res is None


# ------------------------
# Hausdorff distance tests
# ------------------------


def test_compute_hausdorff_distance_identical_masks():
    mask = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    d = ev.compute_hausdorff_distance(
        mask, mask, voxel_size=(1.0, 1.0), max_distance=np.inf, method="standard"
    )
    assert np.isclose(d, 0.0)


def test_compute_hausdorff_distance_separated_points():
    a = np.zeros((1, 5), dtype=bool)
    b = np.zeros((1, 5), dtype=bool)
    a[0, 0] = True
    b[0, 3] = True  # distance 3 along x

    d_std = ev.compute_hausdorff_distance(
        a, b, voxel_size=(1.0, 1.0), max_distance=np.inf, method="standard"
    )
    assert np.isclose(d_std, 3.0)

    d_mod = ev.compute_hausdorff_distance(
        a, b, voxel_size=(1.0, 1.0), max_distance=np.inf, method="modified"
    )
    # only one distance each direction -> mean == max == 3
    assert np.isclose(d_mod, 3.0)


def test_compute_hausdorff_distance_percentile():
    a = np.zeros((1, 5), dtype=bool)
    b = np.zeros((1, 5), dtype=bool)
    # A has foreground at 0,1; B has at 3,4.
    a[0, 0] = True
    a[0, 1] = True
    b[0, 3] = True
    b[0, 4] = True
    # Distances from each point in A to B:
    # A(0)->3 = 3, A(1)->3 = 2 (closest).
    # Similarly B->A: 3->1=2, 4->1=3.
    # So forward distances [3,2], backward [2,3].
    # 50th percentile is ~2.5 each side.
    d_p50 = ev.compute_hausdorff_distance(
        a,
        b,
        voxel_size=(1.0, 1.0),
        max_distance=np.inf,
        method="percentile",
        percentile=50,
    )
    assert np.isclose(d_p50, 2.5, atol=1e-6)


def test_compute_hausdorff_distance_empty_sets():
    max_distance = 5.0
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros_like(a)
    d = ev.compute_hausdorff_distance(
        a, b, voxel_size=(1.0, 1.0), max_distance=max_distance
    )
    assert np.isclose(d, 0.0)

    a[0, 0] = True
    d2 = ev.compute_hausdorff_distance(
        a, b, voxel_size=(1.0, 1.0), max_distance=max_distance
    )
    assert np.isclose(d2, max_distance)


def test_optimized_hausdorff_distances_per_instance():
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
    label = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    scores = ev.score_instance(label, label, voxel_size=(1.0, 1.0))

    assert np.isclose(scores["accuracy"], 1.0)
    assert np.isclose(scores["hausdorff_distance"], 0.0)
    assert np.isclose(scores["normalized_hausdorff_distance"], 1.0)
    assert np.isclose(scores["combined_score"], 1.0)


def test_score_instance_simple_shift():
    # GT has one instance [0,0] and [0,1]
    truth = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int32)
    # Prediction shifted one voxel to the right: [0,1],[0,2] (and needs renumbering)
    pred = np.array([[0, 2, 2], [0, 0, 0]], dtype=np.int32)
    voxel_size = (1.0, 1.0)
    scores = ev.score_instance(pred, truth, voxel_size)

    # Accuracy should not be 1 but positive
    assert 0.0 < scores["accuracy"] < 1.0
    # Hausdorff distance is 1 (each point moves 1 voxel)
    assert np.isclose(scores["hausdorff_distance"], 1.0)


# ------------------------
# score_semantic tests
# ------------------------


def test_score_semantic_perfect_match():
    truth = np.array([[0, 1], [1, 1]], dtype=float)
    pred = truth.copy()
    scores = ev.score_semantic(pred, truth)
    assert np.isclose(scores["iou"], 1.0)
    assert np.isclose(scores["dice_score"], 1.0)


def test_score_semantic_partial_overlap():
    truth = np.array([[0, 1], [1, 1]], dtype=float)
    # prediction misses one positive voxel
    pred = np.array([[0, 1], [0, 1]], dtype=float)

    scores = ev.score_semantic(pred, truth)

    # manual IoU: TP = 2, FP = 0, FN = 1 -> IoU = 2 / (2+0+1) = 2/3
    assert np.isclose(scores["iou"], 2 / 3)
    # manual Dice: 2TP / (2TP + FP + FN) = 4 / (4 + 0 + 1) = 0.8
    assert np.isclose(scores["dice_score"], 0.8)


def test_score_semantic_no_foreground():
    truth = np.zeros((3, 3), dtype=float)
    pred = np.zeros_like(truth)
    scores = ev.score_semantic(pred, truth)
    assert np.isclose(scores["iou"], 1.0)
    assert np.isclose(scores["dice_score"], 1.0)


# ------------------------
# resize_array tests
# ------------------------


def test_resize_array_pad_and_crop():
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
    assert scores["accuracy"] == 0


def test_missing_volume_score_mixed_labels(tmp_path):
    truth_root = tmp_path / "truth_volume.zarr"
    arr_inst = np.zeros((2, 2, 2), dtype=np.uint8)
    arr_sem = np.zeros((2, 2, 2), dtype=np.uint8)

    _create_simple_volume(truth_root, "crop1", "instance", arr_inst)
    _create_simple_volume(truth_root, "crop1", "sem", arr_sem)

    scores = ev.missing_volume_score(
        truth_volume_path=(truth_root / "crop1").as_posix(),
        instance_classes=["instance"],
    )

    assert set(scores.keys()) == {"instance", "sem"}
    assert scores["instance"]["is_missing"] is True
    assert scores["sem"]["is_missing"] is True
    assert scores["instance"]["accuracy"] == 0.0
    assert scores["sem"]["iou"] == 0.0


# ------------------------
# combine_scores tests
# ------------------------


def test_combine_scores_instance_and_semantic():
    # Two volumes, one instance, one semantic
    scores = {
        "crop1": {
            "instance": {
                "accuracy": 1.0,
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


def test_score_label_instance_integration(monkeypatch, tmp_path):
    """
    Small integration test: zarr truth + pred, dummy TEST_CROPS_DICT entry,
    score_label should return instance metrics consistent with score_instance.
    """
    # Arrange mini truth volume
    crop_name = "crop1"
    label_name = "instance"
    truth_root = tmp_path / "truth.zarr"

    arr3d = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.uint8)
    _create_simple_volume(truth_root, crop_name, label_name, arr3d)

    # matching pred volume with same data
    pred_root = tmp_path / "pred.zarr"
    _create_simple_volume(pred_root, crop_name, label_name, arr3d)

    # monkeypatch TEST_CROPS_DICT for this one crop/label
    dummy_crop = DummyCrop(
        voxel_size=(1.0, 1.0, 1.0), shape=arr3d.shape, translation=(0.0, 0.0, 0.0)
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
    assert np.isclose(results["accuracy"], 1.0)
    assert np.isclose(results["hausdorff_distance"], 0.0)
    assert results["is_missing"] is False


def test_score_submission_debug_serial(monkeypatch, tmp_path):
    """
    End-to-end-ish test:
      - create a truth volume with a single crop and label
      - create matching prediction volume
      - zip it
      - run score_submission in DEBUG mode (serial scoring)
    """
    # ensure DEBUG=True so score_submission uses serial path
    monkeypatch.setattr(ev, "DEBUG", True)

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
    assert np.isclose(scores["label_scores"][label_name]["accuracy"], 1.0)
