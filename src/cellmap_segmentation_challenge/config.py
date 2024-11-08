from upath import UPath

REPO_ROOT = UPath(__file__).parent.parent.parent
SEARCH_PATH = (REPO_ROOT / "data/{dataset}/{dataset}.zarr/recon-1/{name}").path
CROP_NAME = UPath("labels/groundtruth/{crop}/{label}").path
RAW_NAME = UPath("em/fibsem-uint8").path
PREDICTIONS_PATH = (REPO_ROOT / "data/predictions/{dataset}.zarr/{crop}").path
PROCESSED_PATH = (REPO_ROOT / "data/processed/{dataset}.zarr/{crop}").path
SUBMISSION_PATH = (REPO_ROOT / "data/submission.zarr").path
