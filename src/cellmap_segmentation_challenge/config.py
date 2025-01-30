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
GT_S3_BUCKET = "janelia-cosem-datasets"
RAW_S3_BUCKET = "janelia-cosem-datasets"
S3_SEARCH_PATH = "{dataset}/{dataset}.zarr/recon-1/{name}"
S3_CROP_NAME = "groundtruth/{crop}/{label}"
S3_RAW_NAME = "em/fibsem-uint8"
