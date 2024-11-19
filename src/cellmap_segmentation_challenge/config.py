from upath import UPath

REPO_ROOT = UPath(__file__).parent.parent.parent
# BASE_DATA_PATH = UPATH(".").path # Use this for saving to your current working directory
BASE_DATA_PATH = REPO_ROOT / "data"
SEARCH_PATH = (BASE_DATA_PATH / "{dataset}/{dataset}.zarr/recon-1/{name}").path
CROP_NAME = UPath("labels/groundtruth/{crop}/{label}").path
RAW_NAME = UPath("em/fibsem-uint8").path
PREDICTIONS_PATH = (BASE_DATA_PATH / "predictions/{dataset}.zarr/{crop}").path
PROCESSED_PATH = (BASE_DATA_PATH / "processed/{dataset}.zarr/{crop}").path
SUBMISSION_PATH = (BASE_DATA_PATH / "submission.zarr").path
