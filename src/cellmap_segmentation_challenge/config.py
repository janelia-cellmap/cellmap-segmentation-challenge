from upath import UPath

# Local paths
REPO_ROOT = UPath(__file__).parent.parent.parent
# BASE_DATA_PATH = UPATH(".").path # Use this for saving to your current working directory
BASE_DATA_PATH = REPO_ROOT / "data"
SEARCH_PATH = (
    BASE_DATA_PATH / "{dataset}" / "{dataset}.zarr" / "recon-1" / "{name}"
).path
CROP_NAME = UPath("labels/groundtruth/{crop}/{label}").path
RAW_NAME = UPath("em/fibsem-uint8").path
PREDICTIONS_PATH = (BASE_DATA_PATH / "predictions" / "{dataset}.zarr" / "{crop}").path
PROCESSED_PATH = (BASE_DATA_PATH / "processed" / "{dataset}.zarr" / "{crop}").path
SUBMISSION_PATH = (BASE_DATA_PATH / "submission.zarr").path

# s3 paths
GT_S3_BUCKET = "janelia-cellmap-fg5f2y1pl8"
RAW_S3_BUCKET = "janelia-cosem-datasets"
S3_SEARCH_PATH = "{dataset}/{dataset}.zarr/recon-1/{name}"
S3_CROP_NAME = "groundtruth/{crop}/{label}"
S3_RAW_NAME = "em/fibsem-uint8"
