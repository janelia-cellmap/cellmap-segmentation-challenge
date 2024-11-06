from upath import UPath

REPO_ROOT = UPath(__file__).parent.parent.parent
SEARCH_PATH = (REPO_ROOT / "data/{dataset}/{dataset}.zarr/recon-1/{name}").path
CROP_NAME = UPath("labels/groundtruth/{crop}/{label}").path
RAW_NAME = UPath("em/fibsem-uint8").path
