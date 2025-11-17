"""Unit tests for crops functions in cellmap_segmentation_challenge.utils.crops"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from cellmap_segmentation_challenge.utils.crops import (
    TestCropRow,
    CropRow,
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


