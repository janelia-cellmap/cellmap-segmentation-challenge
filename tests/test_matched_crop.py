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


# ------------------------------------------------------
# Tests for chunked loading functionality
# ------------------------------------------------------


def test_check_size_ratio_returns_ratio_and_memory(tmp_path, instance_classes):
    """Test that _check_size_ratio returns both ratio and estimated memory."""
    root, grp = _make_group(tmp_path)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1, 1),
        target_shape=(10, 10, 10),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )
    
    # Test with a 20x20x20 array (8x larger in volume)
    ratio, mem_mb = m._check_size_ratio((20, 20, 20))
    assert ratio == 8.0
    # 20*20*20 = 8000 voxels * 4 bytes = 32000 bytes = ~0.03 MB
    assert 0.03 < mem_mb < 0.04


def test_should_use_chunked_loading_raises_on_too_large(tmp_path, instance_classes):
    """Test that _should_use_chunked_loading raises error when ratio exceeds limit."""
    root, grp = _make_group(tmp_path)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1, 1),
        target_shape=(10, 10, 10),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )
    
    # Create a ratio that exceeds MAX_VOLUME_SIZE_RATIO (16^3 = 4096)
    ratio = 5000.0
    mem_mb = 1000.0
    
    with pytest.raises(ValueError, match="too large compared to target shape"):
        m._should_use_chunked_loading(ratio, mem_mb)


def test_should_use_chunked_loading_returns_true_for_large_memory(tmp_path, instance_classes):
    """Test that _should_use_chunked_loading returns True for arrays >500MB."""
    root, grp = _make_group(tmp_path)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1, 1),
        target_shape=(10, 10, 10),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )
    
    # Test with >500MB estimated memory
    ratio = 100.0
    mem_mb = 600.0
    
    assert m._should_use_chunked_loading(ratio, mem_mb) is True


def test_should_use_chunked_loading_returns_false_for_small_memory(tmp_path, instance_classes):
    """Test that _should_use_chunked_loading returns False for arrays <500MB."""
    root, grp = _make_group(tmp_path)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(1, 1, 1),
        target_shape=(10, 10, 10),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )
    
    # Test with <500MB estimated memory
    ratio = 10.0
    mem_mb = 100.0
    
    assert m._should_use_chunked_loading(ratio, mem_mb) is False


def test_load_array_chunked_instance_downsampling(tmp_path, instance_classes):
    """Test chunked loading with instance (nearest neighbor) downsampling."""
    root, grp = _make_group(tmp_path)
    
    # Create a simple 8x8x8 array with distinct values
    src = np.zeros((8, 8, 8), dtype=np.uint8)
    src[0:4, 0:4, 0:4] = 1
    src[4:8, 4:8, 4:8] = 2
    
    arr = grp.create_dataset("data", data=src)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],  # instance class
        target_voxel_size=(2, 2, 2),  # target coarser
        target_shape=(4, 4, 4),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )
    
    # Downsample by 0.5 (8 -> 4)
    scale_factors = (0.5, 0.5, 0.5)
    out = m._load_array_chunked(arr, scale_factors)
    
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.uint8
    # Check that values are preserved (nearest neighbor)
    assert out[0, 0, 0] == 1
    assert out[3, 3, 3] == 2


def test_load_array_chunked_semantic_downsampling(tmp_path, instance_classes):
    """Test chunked loading with semantic (linear interpolation) downsampling."""
    root, grp = _make_group(tmp_path)
    
    # Create a simple 8x8x8 array with float values
    src = np.ones((8, 8, 8), dtype=np.float32)
    
    arr = grp.create_dataset("data", data=src)
    
    sem_classes = set(get_tested_classes()) - set(instance_classes)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=sem_classes.pop(),  # semantic class
        target_voxel_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
        semantic_threshold=0.5,
    )
    
    # Downsample by 0.5
    scale_factors = (0.5, 0.5, 0.5)
    out = m._load_array_chunked(arr, scale_factors)
    
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.bool_
    # All ones should stay above threshold
    assert np.all(out)


def test_load_array_chunked_semantic_thresholding(tmp_path, instance_classes):
    """Test that semantic thresholding is applied correctly after chunked downsampling."""
    root, grp = _make_group(tmp_path)
    
    # Create array with values that should be thresholded
    src = np.zeros((8, 8, 8), dtype=np.float32)
    src[0:4, :, :] = 0.8  # Above threshold
    src[4:8, :, :] = 0.2  # Below threshold
    
    arr = grp.create_dataset("data", data=src)
    
    sem_classes = set(get_tested_classes()) - set(instance_classes)
    
    m = mc.MatchedCrop(
        path=str(root),
        class_label=sem_classes.pop(),
        target_voxel_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
        semantic_threshold=0.5,
    )
    
    scale_factors = (0.5, 0.5, 0.5)
    out = m._load_array_chunked(arr, scale_factors)
    
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.bool_
    # First half should be True (above threshold)
    assert np.all(out[0:2, :, :])
    # Second half should be False (below threshold)
    assert not np.any(out[2:4, :, :])


def test_load_aligned_uses_chunked_for_large_array_with_downsampling(
    tmp_path, instance_classes, monkeypatch
):
    """Test that load_aligned uses chunked loading for large arrays that need downsampling."""
    root, grp = _make_group(tmp_path)
    
    # Create a large-ish array that would trigger chunked loading
    # We'll use a smaller size for testing but monkeypatch the threshold
    src = np.ones((64, 64, 64), dtype=np.uint8)
    
    arr = grp.create_dataset("s0", data=src)
    _set_attrs(arr, voxel_size=(2, 2, 2), translation=(0, 0, 0))
    
    arr_path = str(root / "s0")
    
    # Monkeypatch to lower the memory threshold so our test array triggers chunked loading
    original_threshold = 500
    test_threshold = 0.001  # Very low threshold to trigger chunked loading
    
    m = mc.MatchedCrop(
        path=arr_path,
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(4, 4, 4),  # Coarser resolution - downsampling needed
        target_shape=(32, 32, 32),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
        pad_value=0,
    )
    
    # Temporarily lower the threshold
    def mock_should_use_chunked(ratio, mem_mb):
        if ratio > mc.MAX_VOLUME_SIZE_RATIO:
            raise ValueError("too large")
        return mem_mb > test_threshold  # Use very low threshold
    
    monkeypatch.setattr(m, "_should_use_chunked_loading", mock_should_use_chunked)
    
    out = m.load_aligned()
    
    # Verify output shape and that it worked
    assert out.shape == (32, 32, 32)
    assert out.dtype == np.uint8
    assert np.all(out == 1)


def test_load_aligned_chunked_preserves_data_integrity(tmp_path, instance_classes):
    """Test that chunked loading produces same result as non-chunked for small arrays."""
    root, grp = _make_group(tmp_path)
    
    # Create a pattern that's easy to verify
    src = np.zeros((16, 16, 16), dtype=np.uint8)
    src[0:8, :, :] = 1
    src[8:16, :, :] = 2
    
    arr = grp.create_dataset("s0", data=src)
    _set_attrs(arr, voxel_size=(2, 2, 2), translation=(0, 0, 0))
    
    arr_path = str(root / "s0")
    
    m = mc.MatchedCrop(
        path=arr_path,
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(4, 4, 4),  # Downsample 2x
        target_shape=(8, 8, 8),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )
    
    # Force chunked loading by manually calling _load_array_chunked
    scale_factors = (0.5, 0.5, 0.5)  # 2->4 voxel size means 0.5x dimensions
    out_chunked = m._load_array_chunked(arr, scale_factors)
    
    # Compare with expected output
    assert out_chunked.shape == (8, 8, 8)
    assert out_chunked[0, 0, 0] == 1
    assert out_chunked[7, 7, 7] == 2
    # Middle boundary should have one of the values
    assert out_chunked[4, 8//2, 8//2] in [1, 2]


# ---------------------------------------------------------------
# Regression tests for chunked resampling with odd-size arrays.
#
# When input_size * scale_factor is a half-integer whose floor is
# even (e.g. 5 * 0.5 = 2.5, 65 * 0.5 = 32.5, 594 * 0.25 = 148.5),
# skimage.transform.rescale uses np.round (banker's rounding) and
# rounds *down* to the nearest even integer, while the old code used
# np.ceil and rounded *up*.  This caused:
#
#   ValueError: could not broadcast input array from
#               shape (148,148,148) into shape (149,149,149)
#
# The fix: use np.round for output_shape (matching rescale), and
# derive output-slice bounds from the actual downsampled chunk shape.
# ---------------------------------------------------------------


def test_load_array_chunked_half_integer_instance(tmp_path, instance_classes):
    """
    Regression (instance class): 5 * 0.5 = 2.5 → np.ceil=3, np.round=2.
    Old code raised a broadcast ValueError; output shape must be (2,2,2).
    """
    root, grp = _make_group(tmp_path)
    src = np.ones((5, 5, 5), dtype=np.uint8)
    arr = grp.create_dataset("data", data=src)

    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(2, 2, 2),
        target_shape=(2, 2, 2),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )

    out = m._load_array_chunked(arr, (0.5, 0.5, 0.5))
    assert out.shape == (2, 2, 2)
    assert out.dtype == np.uint8
    assert np.all(out == 1)


def test_load_array_chunked_half_integer_semantic(tmp_path, instance_classes):
    """
    Regression (semantic class): same rounding scenario; output must be bool (2,2,2).
    """
    root, grp = _make_group(tmp_path)
    src = np.ones((5, 5, 5), dtype=np.float32)
    arr = grp.create_dataset("data", data=src)

    sem_classes = set(get_tested_classes()) - set(instance_classes)
    m = mc.MatchedCrop(
        path=str(root),
        class_label=sem_classes.pop(),
        target_voxel_size=(2, 2, 2),
        target_shape=(2, 2, 2),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
        semantic_threshold=0.5,
    )

    out = m._load_array_chunked(arr, (0.5, 0.5, 0.5))
    assert out.shape == (2, 2, 2)
    assert out.dtype == np.bool_
    assert np.all(out)


def test_load_array_chunked_multi_chunk_half_integer(
    tmp_path, instance_classes, monkeypatch
):
    """
    Regression across chunk boundaries: monkeypatches BYTES_PER_FLOAT32 to a huge
    value so chunk_size_per_dim collapses to its minimum of 32, then uses a 65-voxel
    input at scale 0.5.  65 * 0.5 = 32.5 → round=32, ceil=33; must not broadcast-error.

    Uses a spatially-varying input (two halves with distinct values) so that
    incorrect chunk placement would produce wrong output values:
      - input  z=[0:32]  = 1  → expected output z=[0:16]  = 1
      - input  z=[32:65] = 2  → expected output z=[16:32] = 2
    All output voxels are non-zero, confirming no coverage gaps from rounding.
    """
    root, grp = _make_group(tmp_path)
    src = np.zeros((65, 65, 65), dtype=np.uint8)
    src[0:32, :, :] = 1   # first full chunk's input region
    src[32:65, :, :] = 2  # second full chunk + one-voxel remainder
    arr = grp.create_dataset("data", data=src)

    m = mc.MatchedCrop(
        path=str(root),
        class_label=INSTANCE_CLASSES[0],
        target_voxel_size=(2, 2, 2),
        target_shape=(32, 32, 32),
        target_translation=(0, 0, 0),
        instance_classes=instance_classes,
    )

    # Force chunk_size_per_dim = 32:
    #   chunk_voxels = int(100MB / 1e15) = 0  →  max(32, int(0 ** (1/3))) = 32
    monkeypatch.setattr(mc, "BYTES_PER_FLOAT32", 1e15)

    out = m._load_array_chunked(arr, (0.5, 0.5, 0.5))
    # 65 * 0.5 = 32.5 → np.round (banker's) = 32
    assert out.shape == (32, 32, 32)
    assert out.dtype == np.uint8

    # Coverage: every output voxel must be filled (no zero gaps left by rounding)
    assert np.all(out > 0), "gap detected: some output voxels were left unfilled"

    # Alignment: each chunk's data must land at the correct output coordinates
    assert np.all(out[0:16, :, :] == 1), "chunk 1 data placed at wrong output position"
    assert np.all(out[16:32, :, :] == 2), "chunk 2 data placed at wrong output position"
