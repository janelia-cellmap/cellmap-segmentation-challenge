import os
from upath import UPath

# Local paths
REPO_ROOT = UPath(__file__).parent.parent.parent
BASE_DATA_PATH = REPO_ROOT / "data"

SEARCH_PATH = os.path.normpath(
    str(BASE_DATA_PATH / "{dataset}/{dataset}.zarr/recon-1/{name}")
)
CROP_NAME = os.path.normpath("labels/groundtruth/{crop}/{label}")
RAW_NAME = os.path.normpath("em/fibsem-uint8")
PREDICTIONS_PATH = os.path.normpath(
    str(BASE_DATA_PATH / "predictions/{dataset}.zarr/{crop}")
)
PROCESSED_PATH = os.path.normpath(
    str(BASE_DATA_PATH / "processed/{dataset}.zarr/{crop}")
)
SUBMISSION_PATH = os.path.normpath(str(BASE_DATA_PATH / "submission.zarr"))

TRUTH_PATH = (BASE_DATA_PATH / "ground_truth.zarr").path

# s3 paths
# GT_S3_BUCKET = "janelia-cellmap-fg5f2y1pl8"
GT_S3_BUCKET = "janelia-cosem-datasets"
RAW_S3_BUCKET = "janelia-cosem-datasets"
S3_SEARCH_PATH = "{dataset}/{dataset}.zarr/recon-1/{name}"
S3_CROP_NAME = "labels/groundtruth/{crop}/{label}"
S3_RAW_NAME = "em/fibsem-uint8"

INSTANCE_CLASSES = [
    "nuc",
    "vim",
    "ves",
    "endo",
    "lyso",
    "ld",
    "perox",
    "mito",
    "np",
    "mt",
    "cell",
    "instance",
]


VIS_SEARCH_PATHS = {
    "gt": SEARCH_PATH.format(dataset="{dataset}", name=CROP_NAME),
    "predictions": (UPath(PREDICTIONS_PATH) / "{label}").path,
    "processed": (UPath(PROCESSED_PATH) / "{label}").path,
    # "submission": (UPath(SUBMISSION_PATH) / "{crop}" / "{label}").path,
}
