from dataclasses import dataclass
import os

import fsspec
import numpy as np
from upath import UPath
from typing_extensions import Self
from yarl import URL


# get constants from environment, falling back to defaults as needed
TEST_CROP_MANIFEST_URL = os.environ.get(
    "CSC_TEST_CROP_MANIFEST_URL",
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/src/cellmap_segmentation_challenge/utils/test_crop_manifest.csv",
)

MANIFEST_URL = os.environ.get(
    "CSC_FETCH_DATA_MANIFEST_URL",
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/src/cellmap_segmentation_challenge/utils/manifest.csv",
)

ZIP_MANIFEST_URL = os.environ.get(
    "CSC_FETCH_ZIP_DATA_MANIFEST_URL",
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/src/cellmap_segmentation_challenge/utils/zip_manifest.csv",
)


def fetch_manifest(
    url: str | URL,
    file_name: str,
    object: Self,
) -> tuple[str, ...]:
    local_path = UPath(__file__).parent / file_name
    # Attempt to download the manifest file
    try:
        # Get the filesystem and path
        fs, path = fsspec.url_to_fs(str(url))

        # Open the file using the filesystem and save locally
        with fs.open(path, "rb") as src:
            content = src.read()
            if not content:
                raise ValueError("Downloaded manifest is empty.")
        # Only write if content differs from existing file
        write_content = True
        if local_path.exists():
            try:
                with open(local_path, "rb") as existing:
                    existing_content = existing.read()
                if existing_content == content:
                    write_content = False
            except Exception:
                pass
        if write_content:
            with open(local_path, "wb") as dst:
                dst.write(content)
    except Exception:
        if local_path.exists():
            print(
                f"Failed to download manifest file from {url}, using local file {local_path}."
            )
        else:
            raise FileNotFoundError(
                f"Failed to download manifest file from {url} and no local file exists."
            )

    with open(local_path, "r") as f:
        lines = f.read().splitlines()

    if not lines:
        raise ValueError(f"Manifest file at {local_path} is empty.")

    head, *rows = lines
    return tuple(object.from_csv_row(row) for row in rows)


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
    return fetch_manifest(url, "test_crop_manifest.csv", TestCropRow)


@dataclass
class ZipDatasetRow:
    """A dataclass representing a row in the zip dataset manifest file."""

    all_res: bool
    padding: int
    name: str
    url: URL

    @classmethod
    def from_csv_row(cls, row: str) -> Self:
        """Create a CropRow object from a CSV row."""
        all_res, padding, name, url = row.split(",")
        all_res = all_res == "True"
        padding = int(padding)
        return cls(all_res, padding, name, URL(url))


def fetch_zip_manifest(url: str | URL = ZIP_MANIFEST_URL) -> tuple[ZipDatasetRow, ...]:
    """
    Fetch a manifest file from a URL and return a tuple of ZipDatasetRow objects.

    Parameters
    ----------
    url : str or yarl.URL
        The URL to the manifest file.

    Returns
    -------
    tuple[ZipDatasetRow, ...]
        A tuple of ZipDatasetRow objects.
    """
    return fetch_manifest(url, "zip_manifest.csv", ZipDatasetRow)


@dataclass
class CropRow:
    """A dataclass representing a row in the crop manifest file."""

    id: int
    dataset: str
    alignment: str
    gt_source: URL | TestCropRow
    em_url: URL

    @classmethod
    def from_csv_row(cls, row: str) -> Self:
        """Create a CropRow object from a CSV row."""
        id, dataset, alignment, gt_source, em_url = row.split(",")
        return cls(int(id), dataset, alignment, URL(gt_source), URL(em_url))


def fetch_crop_manifest(url: str | URL = MANIFEST_URL) -> tuple[CropRow, ...]:
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
    return fetch_manifest(url, "manifest.csv", CropRow)


TEST_CROPS = fetch_test_crop_manifest()
TEST_CROPS_DICT = {(crop.id, crop.class_label): crop for crop in TEST_CROPS}


def get_test_crops() -> tuple[CropRow, ...]:
    _test_crops = fetch_test_crop_manifest()
    dataset_em_meta = {
        crop.dataset: {"em_url": crop.em_url, "alignment": crop.alignment}
        for crop in fetch_crop_manifest()
    }
    test_crops = []
    test_crop_meta_by_id = {}
    for test_crop in _test_crops:
        if test_crop.id in test_crop_meta_by_id:
            # Make sure metadata for highest resolution, smallest offset, and largest shape is kept
            listed = test_crop_meta_by_id[test_crop.id]
            new_voxel_size = (
                min(l_vs, t_vs)
                for l_vs, t_vs in zip(listed.voxel_size, test_crop.voxel_size)
            )
            new_translation = (
                min(l_trans, t_trans)
                for l_trans, t_trans in zip(listed.translation, test_crop.translation)
            )
            new_shape = (
                max(l_shape, t_shape)
                for l_shape, t_shape in zip(listed.shape, test_crop.shape)
            )
            new_test_crop = TestCropRow(
                test_crop.id,
                test_crop.dataset,
                "test",
                tuple(new_voxel_size),
                tuple(new_translation),
                tuple(new_shape),
            )
            test_crop_meta_by_id[test_crop.id] = new_test_crop
        else:
            test_crop_meta_by_id[test_crop.id] = test_crop

    for id, test_crop in test_crop_meta_by_id.items():
        new_crop = CropRow(
            id,
            test_crop.dataset,
            dataset_em_meta[test_crop.dataset]["alignment"],
            test_crop,
            dataset_em_meta[test_crop.dataset]["em_url"],
        )
        test_crops.append(new_crop)
    return tuple(test_crops)


if __name__ == "__main__":
    sizes = {}
    for crop in TEST_CROPS:
        if crop.id not in sizes:
            sizes[crop.id] = np.prod(crop.shape)
    total_size = np.sum(list(sizes.values())) / 1e9
    print(f"Rough total size of raw data for test crops is {total_size} GB")
