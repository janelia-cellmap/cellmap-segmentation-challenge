
from dataclasses import dataclass
import fsspec
from yarl import URL
from typing_extensions import Self

@dataclass
class CropRow:
    id: int
    dataset: str
    alignment: str
    gt_url: URL
    em_url: URL

    @classmethod
    def from_csv_row(cls, row: str) -> type[Self]:
        id, dataset, alignment, gt_url, em_url = row.split(',')
        return cls(int(id), dataset, alignment, URL(gt_url), URL(em_url))


def fetch_manifest(url: str | URL) -> tuple[CropRow, ...]:
    fs, path = fsspec.url_to_fs(str(url))
    head, *rows = fs.cat_file(path).decode().splitlines()
    return tuple(CropRow.from_csv_row(row) for row in rows)