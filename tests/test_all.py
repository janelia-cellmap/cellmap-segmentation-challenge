# %%
import pytest
import os
from upath import UPath

from cellmap_segmentation_challenge.utils import (
    download_file,
)

from cellmap_segmentation_challenge import RAW_NAME, CROP_NAME

ERROR_TOLERANCE = 0.1


@pytest.fixture(autouse=True)
def reset_env():
    original_env = os.environ.copy()
    # Set the manifest URL for the test crops
    os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)


# %%
@pytest.fixture(scope="session")
def setup_temp_path(tmp_path_factory):
    # temp_dir = (REPO_ROOT / "tests" / "tmp").absolute()  # For debugging
    temp_dir = tmp_path_factory.mktemp("shared_test_dir")

    REPO_ROOT = UPath(temp_dir)
    os.makedirs(REPO_ROOT / "data", exist_ok=True)
    BASE_DATA_PATH = REPO_ROOT / "data"
    SEARCH_PATH = os.path.normpath(
        str(BASE_DATA_PATH / "{dataset}/{dataset}.zarr/recon-1/{name}")
    )
    PREDICTIONS_PATH = os.path.normpath(
        str(BASE_DATA_PATH / "predictions/{dataset}.zarr/{crop}")
    )
    PROCESSED_PATH = os.path.normpath(
        str(BASE_DATA_PATH / "processed/{dataset}.zarr/{crop}")
    )
    SUBMISSION_PATH = os.path.normpath(str(BASE_DATA_PATH / "submission.zarr"))

    yield REPO_ROOT, BASE_DATA_PATH, SEARCH_PATH, PREDICTIONS_PATH, PROCESSED_PATH, SUBMISSION_PATH


@pytest.mark.dependency()
def test_fetch_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    fetch_data_cli.callback(
        crops="test",
        raw_padding=2,
        dest=BASE_DATA_PATH.path,
        access_mode="append",
        fetch_all_em_resolutions=False,
        batch_size=256,
        num_workers=32,
        use_zip=False,
    )


# %%
@pytest.mark.dependency()
def test_fetch_data(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    fetch_data_cli.callback(
        crops="116,118",
        raw_padding=0,
        dest=BASE_DATA_PATH.path,
        access_mode="append",
        fetch_all_em_resolutions=False,
        batch_size=256,
        num_workers=32,
        use_zip=False,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_train(setup_temp_path):
    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/train_config.py",
        REPO_ROOT / "train_config.py",
    )

    from cellmap_segmentation_challenge.cli import train_cli
    from cellmap_segmentation_challenge.utils import make_datasplit_csv

    if (REPO_ROOT / "datasplit.csv").exists():
        (REPO_ROOT / "datasplit.csv").unlink()

    make_datasplit_csv(
        classes=["mito", "er"],
        search_path=SEARCH_PATH,
        csv_path=REPO_ROOT / "datasplit.csv",
        validation_prob=0.0,
    )

    # Now set one of the datasets to "validate" so that we have a validation set
    with open(REPO_ROOT / "datasplit.csv", "r") as f:
        lines = f.readlines()
    with open(REPO_ROOT / "datasplit.csv", "w") as f:
        for i, line in enumerate(lines):
            if i == 0:
                parts = line.strip().split(",")
                parts[0] = '"validate"'
                f.write(",".join(parts) + "\n")
            else:
                f.write(line)

    train_cli.callback(REPO_ROOT / "train_config.py")


# %%
@pytest.mark.dependency(depends=["test_train"])
def test_predict(setup_temp_path):
    from cellmap_segmentation_challenge.cli import predict_cli

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/train_config.py",
        REPO_ROOT / "train_config.py",
    )

    predict_cli.callback(
        REPO_ROOT / "train_config.py",
        crops="116",
        output_path=PREDICTIONS_PATH,
        skip_orthoplanes=True,
        overwrite=True,
        search_path=SEARCH_PATH,
        raw_name=RAW_NAME,
        crop_name=CROP_NAME,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_test_crops"])
def test_predict_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import predict_cli

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/train_config.py",
        REPO_ROOT / "train_config.py",
    )

    predict_cli.callback(
        REPO_ROOT / "train_config.py",
        crops="test",
        output_path=PREDICTIONS_PATH,
        skip_orthoplanes=True,
        overwrite=True,
        search_path=SEARCH_PATH,
        raw_name=RAW_NAME,
        crop_name=CROP_NAME,
    )


# %%
@pytest.mark.dependency(depends=["test_predict"])
def test_process(setup_temp_path):
    from cellmap_segmentation_challenge.cli import process_cli

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/process_config.py",
        REPO_ROOT / "process_config.py",
    )

    process_cli.callback(
        REPO_ROOT / "process_config.py",
        crops="test",
        overwrite=True,
        input_path=PREDICTIONS_PATH,
        output_path=PROCESSED_PATH,
        device="cpu",
        max_workers=os.cpu_count(),
    )


# %%
@pytest.mark.dependency(depends=["test_process"])
def test_pack_results(setup_temp_path):
    from cellmap_segmentation_challenge.cli import package_submission_cli

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    package_submission_cli.callback(
        PROCESSED_PATH, SUBMISSION_PATH, overwrite=True, max_workers=os.cpu_count()
    )
