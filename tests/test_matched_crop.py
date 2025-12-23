# tests/test_matched_crop.py
import numpy as np
import pytest
import zarr

# ⚠️ Adjust this import to wherever you put the code:
# e.g. from cellmap_segmentation_challenge.matched_crop import ...
from cellmap_segmentation_challenge.utils import matched_crop as mc
from cellmap_segmentation_challenge.utils import get_tested_classes
from cellmap_segmentation_challenge.config import INSTANCE_CLASSES


@pytest.fixture
def instance_classes():
    return INSTANCE_CLASSES[:2]


def _make_group(tmp_path, name="vol.zarr"):
    root = tmp_path / name
    grp = zarr.open_group(str(root), mode="w")
    return root, grp


def _set_attrs(arr, voxel_size=None, translation=None):
    if voxel_size is not None:
        arr.attrs["voxel_size"] = tuple(voxel_size)
    if translation is not None:
        arr.attrs["translation"] = tuple(translation)


# --------------------------
# Small unit tests: helpers
# --------------------------


def test_get_attr_any():
    attrs = {"voxel_size": (4, 4), "foo": 1}
    assert mc._get_attr_any(attrs, ["missing", "voxel_size"]) == (4, 4)
    assert mc._get_attr_any(attrs, ["missing1", "missing2"]) is None


def test_parse_voxel_size_and_translation():
    attrs = {"resolution": (2, 3), "offset": (10, 20)}
    assert mc._parse_voxel_size(attrs) == (2.0, 3.0)
    assert mc._parse_translation(attrs) == (10.0, 20.0)

    assert mc._parse_voxel_size({}) is None
    assert mc._parse_translation({}) is None


def test_resize_pad_crop_center_pad():
    img = np.ones((2, 2), dtype=np.uint8)
    out = mc._resize_pad_crop(img, target_shape=(4, 4), pad_value=0)

    assert out.shape == (4, 4)
    # original 2x2 should land centered: indices [1:3, 1:3]
    assert np.all(out[1:3, 1:3] == 1)
    # borders should be pad_value
    assert np.all(out[[0, 3], :] == 0)
    assert np.all(out[:, [0, 3]] == 0)


def test_resize_pad_crop_center_crop():
    img = np.arange(25, dtype=np.int32).reshape(5, 5)
    out = mc._resize_pad_crop(img, target_shape=(3, 3), pad_value=0)

    assert out.shape == (3, 3)
    # center crop of 5 -> 3 is indices [1:4, 1:4]
    assert np.array_equal(out, img[1:4, 1:4])


# ------------------------------------------
# _select_non_ome_level selection heuristic
# ------------------------------------------


def test_select_non_ome_level_prefers_not_finer_and_closest(tmp_path, instance_classes):
    root, grp = _make_group(tmp_path)

    # target voxel size = (2,2)
    a0 = grp.create_dataset("s0", data=np.zeros((4, 4), dtype=np.uint8))
    _set_attrs(a0, voxel_size=(1, 1), translation=(0, 0))  # finer (penalized)

    a1 = grp.create_dataset("s1", data=np.zeros((2, 2), dtype=np.uint8))
    _set_attrs(a1, voxel_size=(2, 2), translation=(0, 0))  # perfect match

    a2 = grp.create_dataset("s2", data=np.zeros((1, 1), dtype=np.uint8))
    _set_attrs(a2, voxel_size=(4, 4), translation=(0, 0))  # coarser but farther

    m = mc.MatchedCrop(
        path=str(root),
        class_label="mito",
        target_voxel_size=(2, 2),
        target_shape=(2, 2),
        target_translation=(0, 0),
        instance_classes=instance_classes,
    )

    key, vs, tr = m._select_non_ome_level(grp)
    assert key == "s1"
    assert vs == (2.0, 2.0)
    assert tr == (0.0, 0.0)


def test_select_non_ome_level_handles_missing_voxel_size_attrs(
    tmp_path, instance_classes
):
    root, grp = _make_group(tmp_path)

    a0 = grp.create_dataset("a", data=np.zeros((2, 2), dtype=np.uint8))
    # no attrs on a0
    a1 = grp.create_dataset("b", data=np.zeros((2, 2), dtype=np.uint8))
    _set_attrs(a1, voxel_size=(3, 3), translation=(0, 0))

    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(2, 2),
        target_shape=(2, 2),
        target_translation=(0, 0),
        instance_classes=instance_classes,
    )
    key, vs, tr = m._select_non_ome_level(grp)
    # Should prefer the one with real voxel_size over unknown
    assert key == "b"
    assert vs == (3.0, 3.0)
    assert tr == (0.0, 0.0)


# ---------------------------------------
# load_aligned: non-OME group multiscale
# ---------------------------------------


def test_load_aligned_instance_resample_nearest_non_ome_group(
    tmp_path, instance_classes
):
    """
    Instance: order=0 rescale; check label integrity under upsampling.
    """
    root, grp = _make_group(tmp_path)

    # Provide a single-scale array at voxel_size=(2,2) that must be rescaled to target (1,1)
    # scale_factors = in_vs / tgt_vs = (2,2)/(1,1) => upsample x2 in each dim
    src = np.array([[0, 1], [2, 0]], dtype=np.uint8)

    a = grp.create_dataset("s1", data=src)
    _set_attrs(a, voxel_size=(2, 2), translation=(0, 0))

    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],  # instance
        target_voxel_size=(1, 1),
        target_shape=(4, 4),
        target_translation=(0, 0),
        instance_classes=instance_classes,
        pad_value=0,
    )

    out = m.load_aligned()
    assert out.shape == (4, 4)
    # Nearest upsample should replicate 2x2 blocks
    # expected:
    # [[0,0,1,1],
    #  [0,0,1,1],
    #  [2,2,0,0],
    #  [2,2,0,0]]
    expected = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]], dtype=np.uint8
    )
    assert np.array_equal(out, expected)


def test_load_aligned_semantic_resample_linear_and_threshold(
    tmp_path, instance_classes
):
    """
    Semantic: order=1 rescale + threshold => boolean output.
    Use constant input so interpolation is deterministic.
    """
    root, grp = _make_group(tmp_path)

    src = np.ones(
        (2, 2), dtype=np.float32
    )  # all ones should stay > threshold after rescale
    a = grp.create_dataset("s1", data=src)
    _set_attrs(a, voxel_size=(2, 2), translation=(0, 0))

    sem_classes = set(get_tested_classes()) - set(instance_classes)
    m = mc.MatchedCrop(
        path=str(root),
        class_label=sem_classes.pop(),  # semantic (not in instance_classes)
        target_voxel_size=(1, 1),
        target_shape=(4, 4),
        target_translation=(0, 0),
        instance_classes=instance_classes,
        semantic_threshold=0.5,
        pad_value=0,
    )

    out = m.load_aligned()
    assert out.shape == (4, 4)
    assert out.dtype == np.bool_
    assert np.all(out)  # all True


def test_load_aligned_semantic_thresholding_zeros(tmp_path, instance_classes):
    root, grp = _make_group(tmp_path)

    src = np.zeros((2, 2), dtype=np.float32)
    a = grp.create_dataset("s1", data=src)
    _set_attrs(a, voxel_size=(2, 2), translation=(0, 0))

    m = mc.MatchedCrop(
        path=str(root),
        class_label="membrane",
        target_voxel_size=(1, 1),
        target_shape=(4, 4),
        target_translation=(0, 0),
        instance_classes=instance_classes,
        semantic_threshold=0.5,
        pad_value=0,
    )

    out = m.load_aligned()
    assert out.dtype == np.bool_
    assert not np.any(out)  # all False


# ---------------------------------------
# load_aligned: single-scale array + shift
# ---------------------------------------


def test_load_aligned_translation_positive_shift(tmp_path, instance_classes):
    """
    If input_translation > target_translation, rel is positive and output is padded at start
    (source content shifts forward in that axis).
    """
    root, grp = _make_group(tmp_path)

    src = np.zeros((5, 5), dtype=np.uint8)
    src[0, 0] = 9

    arr = grp.create_dataset("s0", data=src)
    _set_attrs(
        arr, voxel_size=(1, 1), translation=(2, 0)
    )  # shift along axis0 by +2 voxels

    # Point MatchedCrop directly at the array node to exercise zarr.Array branch:
    arr_path = str(root / "s0")

    m = mc.MatchedCrop(
        path=arr_path,
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1),
        target_shape=(5, 5),
        target_translation=(0, 0),
        instance_classes=instance_classes,
        pad_value=0,
    )

    out = m.load_aligned()
    assert out.shape == (5, 5)

    # Source (0,0) should land at (2,0) after positive rel shift along axis0
    assert out[2, 0] == 9
    # Original location should be zero
    assert out[0, 0] == 0


def test_load_aligned_translation_negative_shift(tmp_path, instance_classes):
    """
    If input_translation < target_translation, rel is negative and input is cropped at start
    (source content shifts backward).
    """
    root, grp = _make_group(tmp_path)

    src = np.zeros((5, 5), dtype=np.uint8)
    src[4, 0] = 7

    arr = grp.create_dataset("s0", data=src)
    _set_attrs(arr, voxel_size=(1, 1), translation=(0, 0))

    arr_path = str(root / "s0")

    m = mc.MatchedCrop(
        path=arr_path,
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1),
        target_shape=(5, 5),
        target_translation=(2, 0),  # target starts later => rel negative
        instance_classes=instance_classes,
        pad_value=0,
    )

    out = m.load_aligned()
    assert out.shape == (5, 5)

    # With target_translation=(2,0), rel ~ (0 - 2)//1 = -2 => crop first 2 rows of input.
    # src[4,0] moves to out[2,0]
    assert out[2, 0] == 7


# ------------------------------------------------
# load_aligned: fallback pad/crop when no voxel_size
# ------------------------------------------------


def test_load_aligned_fallback_center_pad_crop_when_no_voxel_size(
    tmp_path, instance_classes
):
    root, grp = _make_group(tmp_path)

    src = np.ones((2, 2), dtype=np.uint8)
    arr = grp.create_dataset("s0", data=src)
    # no voxel_size attrs => triggers fallback path only if shape mismatch

    arr_path = str(root / "s0")

    m = mc.MatchedCrop(
        path=arr_path,
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1),  # unused in fallback
        target_shape=(4, 4),
        target_translation=(0, 0),
        instance_classes=instance_classes,
        pad_value=0,
    )

    out = m.load_aligned()
    assert out.shape == (4, 4)
    assert np.all(out[1:3, 1:3] == 1)
    assert np.all(out[[0, 3], :] == 0)
    assert np.all(out[:, [0, 3]] == 0)
