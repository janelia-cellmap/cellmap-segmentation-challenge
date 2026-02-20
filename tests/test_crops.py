"""Unit tests for crops functions in cellmap_segmentation_challenge.utils.crops"""

from cellmap_segmentation_challenge.utils.crops import (
    TestCropRow,
    CropRow,
    get_test_crop_labels,
    fetch_test_crop_manifest,
)


class TestTestCropRowDataclass:
    """Tests for TestCropRow dataclass"""

    def test_from_csv_row_simple(self):
        """Test creating TestCropRow from a simple CSV row"""
        row = "116,jrc_hela-2,er,[4.0;4.0;4.0],[100.0;200.0;300.0],[64;64;64]"
        crop = TestCropRow.from_csv_row(row)

        assert crop.id == 116
        assert crop.dataset == "jrc_hela-2"
        assert crop.class_label == "er"
        assert crop.voxel_size == (4.0, 4.0, 4.0)
        assert crop.translation == (100.0, 200.0, 300.0)
        assert crop.shape == (64, 64, 64)

    def test_from_csv_row_different_shapes(self):
        """Test creating TestCropRow with different shape dimensions"""
        row = "234,jrc_cos7-1a,nuc,[8.0;8.0;8.0],[1000.0;2000.0;3000.0],[128;128;128]"
        crop = TestCropRow.from_csv_row(row)

        assert crop.id == 234
        assert crop.dataset == "jrc_cos7-1a"
        assert crop.class_label == "nuc"
        assert crop.voxel_size == (8.0, 8.0, 8.0)
        assert crop.translation == (1000.0, 2000.0, 3000.0)
        assert crop.shape == (128, 128, 128)

    def test_from_csv_row_float_voxel_sizes(self):
        """Test creating TestCropRow with non-integer voxel sizes"""
        row = "118,jrc_hela-3,mito,[2.5;2.5;2.5],[50.0;100.0;150.0],[32;32;32]"
        crop = TestCropRow.from_csv_row(row)

        assert crop.id == 118
        assert crop.voxel_size == (2.5, 2.5, 2.5)

    def test_from_csv_row_negative_translations(self):
        """Test creating TestCropRow with negative translations"""
        row = "120,jrc_hela-2,cell,[-4.0;4.0;4.0],[-100.0;200.0;300.0],[64;64;64]"
        crop = TestCropRow.from_csv_row(row)

        assert crop.voxel_size == (-4.0, 4.0, 4.0)
        assert crop.translation == (-100.0, 200.0, 300.0)


class TestCropRowDataclass:
    """Tests for CropRow dataclass"""

    def test_from_csv_row_simple(self):
        """Test creating CropRow from a simple CSV row"""
        row = "116,jrc_hela-2,xy,s3://janelia-cellmap-0/jrc_hela-2/jrc_hela-2.n5/groundtruth,s3://janelia-cellmap-0/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint8"
        crop = CropRow.from_csv_row(row)

        assert crop.id == 116
        assert crop.dataset == "jrc_hela-2"
        assert crop.alignment == "xy"
        assert str(crop.gt_source).startswith("s3://janelia-cellmap-0/jrc_hela-2")
        assert str(crop.em_url).startswith("s3://janelia-cellmap-0/jrc_hela-2")

    def test_from_csv_row_different_alignment(self):
        """Test creating CropRow with different alignment"""
        row = "234,jrc_cos7-1a,xz,s3://janelia-cellmap-0/jrc_cos7-1a/jrc_cos7-1a.n5/groundtruth,s3://janelia-cellmap-0/jrc_cos7-1a/jrc_cos7-1a.n5/em/fibsem-uint8"
        crop = CropRow.from_csv_row(row)

        assert crop.id == 234
        assert crop.dataset == "jrc_cos7-1a"
        assert crop.alignment == "xz"

    def test_from_csv_row_different_datasets(self):
        """Test creating CropRow from different datasets"""
        row = "342,jrc_jurkat-1,xy,s3://janelia-cellmap-0/jrc_jurkat-1/jrc_jurkat-1.n5/groundtruth,s3://janelia-cellmap-0/jrc_jurkat-1/jrc_jurkat-1.n5/em/fibsem-uint8"
        crop = CropRow.from_csv_row(row)

        assert crop.dataset == "jrc_jurkat-1"
        assert crop.id == 342


class TestGetTestCropLabels:
    """Tests for get_test_crop_labels function"""

    def test_get_test_crop_labels_returns_list(self):
        """Test that get_test_crop_labels returns a list"""
        # Get the first crop ID from the manifest
        test_crops = fetch_test_crop_manifest()
        if not test_crops:
            return  # Skip if no test crops available
        first_crop_id = test_crops[0].id
        labels = get_test_crop_labels(first_crop_id)
        assert isinstance(labels, list)

    def test_get_test_crop_labels_returns_correct_labels(self):
        """Test that get_test_crop_labels returns the correct labels for a crop"""
        test_crops = fetch_test_crop_manifest()
        if not test_crops:
            return  # Skip if no test crops available
        
        # Get the first crop ID
        first_crop_id = test_crops[0].id
        
        # Get expected labels by filtering the manifest
        expected_labels = [crop.class_label for crop in test_crops if crop.id == first_crop_id]
        
        # Get actual labels from the function
        actual_labels = get_test_crop_labels(first_crop_id)
        
        # Check that they match
        assert sorted(expected_labels) == sorted(actual_labels)

    def test_get_test_crop_labels_different_crops_different_labels(self):
        """Test that different crops can have different numbers of labels"""
        test_crops = fetch_test_crop_manifest()
        if len(test_crops) < 2:
            return  # Skip if not enough test crops
        
        # Get two different crop IDs
        crop_ids = list(set(crop.id for crop in test_crops))
        if len(crop_ids) < 2:
            return
        
        crop_id_1 = crop_ids[0]
        crop_id_2 = crop_ids[1]
        
        labels_1 = get_test_crop_labels(crop_id_1)
        labels_2 = get_test_crop_labels(crop_id_2)
        
        # Just check they are both lists (they may or may not have same length)
        assert isinstance(labels_1, list)
        assert isinstance(labels_2, list)
        
    def test_get_test_crop_labels_nonexistent_crop(self):
        """Test that get_test_crop_labels returns empty list for nonexistent crop"""
        # Use a very large crop ID that shouldn't exist
        labels = get_test_crop_labels(999999)
        assert labels == []
