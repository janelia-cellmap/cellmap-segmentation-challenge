# %%
import json
import pytest
import shutil
import os
from upath import UPath

import numpy as np
from skimage.transform import rescale
from cellmap_segmentation_challenge.utils import (
    simulate_predictions_accuracy,
    simulate_predictions_iou_binary,
    download_file,
)


from cellmap_segmentation_challenge import RAW_NAME, CROP_NAME
from cellmap_segmentation_challenge.utils.submission import (
    zip_submission,
    save_numpy_class_arrays_to_zarr,
)

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
        validation_prob=0.5,
    )

    train_cli.callback(REPO_ROOT / "train_config.py")


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
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
@pytest.mark.dependency(depends=["test_fetch_data"])
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

    truth_path = REPO_ROOT / "data" / "truth.zarr"

    package_submission_cli.callback(
        PROCESSED_PATH, truth_path.path, overwrite=True, max_workers=os.cpu_count()
    )


# %%
@pytest.mark.parametrize(
    "scale, iou, accuracy",
    [
        (None, None, None),
        (2, None, None),  # 2x resolution
        (0.5, None, None),  # 0.5x resolution
        (None, 0.8, 0.8),  # 0.8 iou, 0.8 accuracy
        (2, 0.8, 0.8),  # 2x resolution, 0.8 iou, 0.8 accuracy
        (0.5, 0.8, 0.8),  # 0.5x resolution, 0.8 iou, 0.8 accuracy
    ],
)
@pytest.mark.dependency(depends=["test_pack_results"])
def test_evaluate(setup_temp_path, scale, iou, accuracy):
    from cellmap_segmentation_challenge.cli import evaluate_cli
    from cellmap_segmentation_challenge.evaluate import INSTANCE_CLASSES
    import zarr

    (
        REPO_ROOT,
        BASE_DATA_PATH,
        SEARCH_PATH,
        PREDICTIONS_PATH,
        PROCESSED_PATH,
        SUBMISSION_PATH,
    ) = setup_temp_path

    truth_path = REPO_ROOT / "data" / "truth.zarr"

    if any([scale, iou, accuracy]):
        submission_path = REPO_ROOT / "data" / "submission.zarr"
        if submission_path.exists():
            # Remove the submission zarr if it already exists
            shutil.rmtree(submission_path)
        submission_zarr = zarr.open(submission_path, mode="w")
        truth_zarr = zarr.open(truth_path, mode="r")
        for crop in truth_zarr.keys():
            crop_zarr = truth_zarr[crop]
            submission_zarr.create_group(crop)
            for label in crop_zarr.keys():
                label_zarr = crop_zarr[label]
                attrs = label_zarr.attrs.asdict()
                truth = label_zarr[:]
                pred = truth.copy()

                if iou is not None and label not in INSTANCE_CLASSES:
                    pred = simulate_predictions_iou_binary(pred, iou)
                if accuracy is not None and label in INSTANCE_CLASSES:
                    pred = simulate_predictions_accuracy(pred, accuracy)

                if scale:
                    pred = rescale(pred, scale, order=0, preserve_range=True)
                    old_voxel_size = attrs["voxel_size"]
                    new_voxel_size = [s / scale for s in attrs["voxel_size"]]
                    attrs["voxel_size"] = new_voxel_size
                    # Adjust the translation
                    attrs["translation"] = [
                        t + (n - o) / 2
                        for t, o, n in zip(
                            attrs["translation"], old_voxel_size, new_voxel_size
                        )
                    ]

                save_numpy_class_arrays_to_zarr(
                    submission_path,
                    crop,
                    [label],
                    [pred],
                    attrs=attrs,
                )
    else:
        submission_path = truth_path
    zip_submission(submission_path)

    evaluate_cli.callback(
        submission_path.with_suffix(".zip"),
        result_file=REPO_ROOT / "result.json",
        truth_path=truth_path,
        instance_classes=",".join(INSTANCE_CLASSES),
    )

    # Check the results:
    with open(REPO_ROOT / "result_submitted_only.json") as f:
        results = json.load(f)

    if iou is None and accuracy is None:
        assert (
            1 - results["overall_score"] < ERROR_TOLERANCE
        ), f"Overall score should be 1 but is: {results['overall_score']}"
    else:
        # Check all accuracy scores and ious
        for label, scores in results["label_scores"].items():
            if label in INSTANCE_CLASSES:
                assert (
                    np.abs((accuracy or 1) - scores["accuracy"]) < ERROR_TOLERANCE
                ), f"Accuracy score for {label} should be {(accuracy or 1)} but is: {scores['accuracy']}"
            else:
                assert (
                    np.abs((iou or 1) - scores["iou"]) < ERROR_TOLERANCE
                ), f"IoU score for {label} should be {(iou or 1)} but is: {scores['iou']}"
