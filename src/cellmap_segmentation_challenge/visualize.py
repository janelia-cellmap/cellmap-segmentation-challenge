from typing import Sequence
import neuroglancer
from glob import glob
from upath import UPath
from . import CROP_NAME, REPO_ROOT, SEARCH_PATH
from .utils.datasplit import get_raw_path, get_dataset_name

def visualize(
    datasets: str | Sequence = "*",
    crops: str | Sequence = "*",
    classes="*",
    datapaths: dict = {"raw": },
):
    # Start Neuroglancer with a local web server
    neuroglancer.set_server_bind_address("0.0.0.0", 8080)

    # Path to your Zarr dataset
    zarr_path = "path/to/your.zarr"  # Replace with the path to your Zarr file

    # Initialize the Neuroglancer viewer
    viewer = neuroglancer.Viewer()

    # Set up the Zarr data layer
    with viewer.txn() as s:
        s.layers["zarr_data"] = neuroglancer.ImageLayer(
            source=zarr_path,
        )

    # Output the viewer URL to open in a browser
    print(f"Neuroglancer viewer running at: {viewer}")
