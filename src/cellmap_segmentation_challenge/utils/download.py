import os
from typing import Sequence

from .shared import TRUTH_DATASETS, RESOLUTION_LEVELS, CLASS_DATASETS


def download(
    labels: Sequence[str],
    resolution: int,
    output_path: str | os.PathLike,
    padding: Sequence[int] = (0, 0, 0),
    fill_value: float = 0,
    format: str = "zarr2",
):
    """Download the dataset for the CellMap Segmentation Challenge.

    Args:
        labels: Sequence[str]
            The label classes to download. e.g. ["cell", "nuc", "mito"]
        resolution : int
            The resolution of the data in nanometers per voxel. This should be one of the following: 8, 16, 32, 64, 128, 256, 512, 1024, 2048. Data should be isotropic.
        output_path: str | os.PathLike
            The path to the directory where to save the dataset.
        padding: Sequence[int], optional
            The amount of padding around ground truth volumes to add to raw data, in voxels per axis. Default is (0, 0, 0)
        fill_value: float, optional
            The value to fill the padding with if it extends past the edge of the available raw data. Default is 0.
        format: str, optional
            The format to save the dataset in. Default is "zarr2". Options are "zarr2" and "hdf5". HDF5 will have smaller chunk sizes, useful for reading 2D slices.
    """

    resolution_level = RESOLUTION_LEVELS[resolution]

    for label in labels:
        for dataset in CLASS_DATASETS[label]:
            # Initialize destination dataset as needed
            # including downloading raw data as specified, if not already downloaded
            ...
            
            # Download/convert the dataset
            path = TRUTH_DATASETS[dataset].format(label=label, resolution_level=resolution_level)
            ...
        
