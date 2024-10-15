from dataclasses import dataclass
import fsspec
from yarl import URL
from typing_extensions import Self


@dataclass
class CropRow:
    """A dataclass representing a row in the crop manifest file."""

    id: int
    dataset: str
    alignment: str
    gt_url: URL
    em_url: URL

    @classmethod
    def from_csv_row(cls, row: str) -> type[Self]:
        """Create a CropRow object from a CSV row."""
        id, dataset, alignment, gt_url, em_url = row.split(",")
        return cls(int(id), dataset, alignment, URL(gt_url), URL(em_url))


def fetch_manifest(url: str | URL) -> tuple[CropRow, ...]:
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
