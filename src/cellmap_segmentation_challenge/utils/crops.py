from dataclasses import dataclass
import os

import fsspec
from typing_extensions import Self
from yarl import URL


# get constants from environment, falling back to defaults as needed
MANIFEST_URL = os.environ.get(
    "CSC_FETCH_DATA_MANIFEST_URL",
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/src/cellmap_segmentation_challenge/utils/manifest.csv",
)


@dataclass
class CropRow:
    """A dataclass representing a row in the crop manifest file."""

    id: int
    dataset: str
    alignment: str
    gt_url: URL
    em_url: URL

    @classmethod
    def from_csv_row(cls, row: str) -> Self:
        """Create a CropRow object from a CSV row."""
        id, dataset, alignment, gt_url, em_url = row.split(",")
        return cls(int(id), dataset, alignment, URL(gt_url), URL(em_url))


def fetch_manifest(url: str | URL = MANIFEST_URL) -> tuple[CropRow, ...]:
    """
    Fetch a manifest file from a URL and return a tuple of CropRow objects.

    Parameters
    ----------
    url : str or yarl.URL
        The URL to the manifest file.

    Returns
    -------
    tuple[CropRow, ...]
        A tuple of CropRow objects.
    """
    fs, path = fsspec.url_to_fs(str(url))
    head, *rows = fs.cat_file(path).decode().splitlines()
    return tuple(CropRow.from_csv_row(row) for row in rows)


# get constants from environment, falling back to defaults as needed
TEST_CROP_MANIFEST_URL = os.environ.get(
    "CSC_TEST_CROP_MANIFEST_URL",
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/src/cellmap_segmentation_challenge/utils/test_crop_manifest.csv",
)


@dataclass
class TestCropRow:
    """A dataclass representing a row in the test crop manifest file."""

    id: int
    dataset: str
    class_label: str
    voxel_size: tuple[float, ...]
    translation: tuple[float, ...]
    shape: tuple[int, ...]

    @classmethod
    def from_csv_row(cls, row: str) -> Self:
        """Create a CropRow object from a CSV row."""
        id, dataset, class_label, voxel_size, translation, shape = row.split(",")
        return cls(
            int(id),
            dataset,
            class_label,
            tuple(map(float, voxel_size.strip("[]").split(";"))),
            tuple(map(float, translation.strip("[]").split(";"))),
            tuple(map(int, shape.strip("[]").split(";"))),
        )


def fetch_test_crop_manifest(
    url: str | URL = TEST_CROP_MANIFEST_URL,
) -> tuple[TestCropRow, ...]:
    """
    Fetch a test manifest file from a URL and return a tuple of TestCropRow objects.

    Parameters
    ----------
    url : str or yarl.URL
        The URL to the manifest file.

    Returns
    -------
    tuple[TestCropRow, ...]
        A tuple of TestCropRow objects.
    """
    fs, path = fsspec.url_to_fs(str(url))
    head, *rows = fs.cat_file(path).decode().splitlines()
    return tuple(TestCropRow.from_csv_row(row) for row in rows)


TEST_CROPS = fetch_test_crop_manifest()
TEST_CROPS_DICT = {(crop.id, crop.class_label): crop for crop in TEST_CROPS}
