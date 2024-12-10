# %%
import json
import pytest
import shutil
import os

import numpy as np
from skimage.measure import label as relabel
from cellmap_segmentation_challenge import REPO_ROOT


# %%
@pytest.fixture(scope="session")
def setup_temp_path(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("shared_test_dir")
    os.environ["TEST_TMP_DIR"] = str(temp_dir)
    yield temp_dir
    # Cleanup: Unset the environment variable after tests are done
    del os.environ["TEST_TMP_DIR"]


# %%
@pytest.mark.dependency()
def test_fetch_data(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    os.makedirs(setup_temp_path / "data", exist_ok=True)
    fetch_data_cli.callback(
        crops="116,118",
        raw_padding=0,
        dest=setup_temp_path / "data",
        access_mode="append",
        fetch_all_em_resolutions=False,
        batch_size=256,
        num_workers=32,
    )


@pytest.mark.dependency()
def test_fetch_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
    )
    workdir = setup_temp_path

    os.makedirs(workdir / "data", exist_ok=True)
    fetch_data_cli.callback(
        crops="test",
        raw_padding=0,
        dest=workdir / "data",
        access_mode="append",
        fetch_all_em_resolutions=False,
        batch_size=256,
        num_workers=32,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_train():
    workdir = os.environ["TEST_TMP_DIR"]
    shutil.copy(REPO_ROOT / "tests" / "train_config.py", workdir)
    shutil.copy(REPO_ROOT / "tests" / "process_config.py", workdir)
    os.chdir(workdir)

    from cellmap_segmentation_challenge.cli import train_cli
    from cellmap_segmentation_challenge.utils import make_datasplit_csv

    make_datasplit_csv(
        classes=["mito", "er"],
        search_path="data/{dataset}/{dataset}.zarr/recon-1/{name}",
        validation_prob=0.5,
    )

    train_cli.callback("train_config.py")


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_predict():
    workdir = os.environ["TEST_TMP_DIR"]
    os.chdir(workdir)
    from cellmap_segmentation_challenge.cli import predict_cli

    PREDICTION_PATH = os.path.join(
        workdir, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )

    predict_cli.callback(
        "train_config.py",
        crops="116",
        output_path=PREDICTION_PATH,
        do_orthoplanes=False,
        overwrite=True,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_test_crops"])
def test_predict_test_crops():
    from cellmap_segmentation_challenge.cli import predict_cli

    workdir = os.environ["TEST_TMP_DIR"]
    PREDICTION_PATH = os.path.join(
        workdir, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )

    predict_cli.callback(
        "train_config.py",
        crops="test",
        output_path=PREDICTION_PATH,
        do_orthoplanes=False,
        overwrite=True,
    )


# %%
@pytest.mark.dependency(depends=["test_predict"])
def test_process():
    from cellmap_segmentation_challenge.cli import process_cli

    os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
    )
    workdir = os.environ["TEST_TMP_DIR"]
    PREDICTION_PATH = os.path.join(
        workdir, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )
    PROCESSED_PATH = os.path.join(
        workdir, *"data/processed/{dataset}.zarr/{crop}".split("/")
    )

    process_cli.callback(
        "process_config.py",
        crops="test",
        overwrite=True,
        input_path=PREDICTION_PATH,
        output_path=PROCESSED_PATH,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_pack_results(setup_temp_path):
    from cellmap_segmentation_challenge.cli import package_submission_cli

    os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
    )
    workdir = setup_temp_path
    PROCESSED_PATH = os.path.join(
        workdir,
        *"data/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/{crop}".split("/"),
    )
    TRUTH_PATH = os.path.join(workdir, "truth.zarr")
    os.environ["CSC_TRUTH_PATH"] = TRUTH_PATH

    package_submission_cli.callback(PROCESSED_PATH, TRUTH_PATH, overwrite=True)


# %%
@pytest.mark.parametrize(
    "scale, iou, accuracy",
    [
        (None, None, None),
        # (2, None, None),
        # (0.5, None, None),
        # (None, 0.8, 0.8),
        # (2, 0.8, 0.8),
        # (0.5, 0.8, 0.8),
    ],
)
@pytest.mark.dependency(depends=["test_pack_results"])
def test_evaluate(scale, iou, accuracy):
    from cellmap_segmentation_challenge.cli import evaluate_cli
    from cellmap_segmentation_challenge.evaluate import INSTANCE_CLASSES

    os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
    )

    workdir = os.environ["TEST_TMP_DIR"]
    TRUTH_PATH = os.environ["CSC_TRUTH_PATH"]

    if any([scale, iou, accuracy]):
        SUBMISSION_PATH = os.path.join(workdir, "submission.zarr")
        if scale:
            resample_arrays(TRUTH_PATH, (scale,) * 3)
    else:
        SUBMISSION_PATH = TRUTH_PATH

    evaluate_cli.callback(
        SUBMISSION_PATH.replace(".zarr", ".zip"),
        result_file=os.path.join(workdir, "result.json"),
        truth_path=TRUTH_PATH,
        instance_classes=",".join(INSTANCE_CLASSES),
    )

    # Check the results:
    with open(os.path.join(workdir, "result_submitted_only.json")) as f:
        results = json.load(f)

    if iou is not None and accuracy is not None:
        assert 1 - results["overall_score"] < 1e-4
    else:
        assert np.abs(iou - results["overall_semantic_score"]) < 1e-4
        # Check all accuracy scores
        for label, scores in results["label_scores"].items():
            if label in INSTANCE_CLASSES:
                assert np.abs(accuracy - scores["accuracy"]) < 1e-4


# %%


def resample_arrays(zarr_path: str, scale_factors: tuple[float, ...] | None = None): ...


def simulate_predictions_iou(true_labels, iou):
    # TODO: Add false positives (only makes false negatives currently)

    pred_labels = np.zeros_like(true_labels)
    for i in np.unique(true_labels):
        if i == 0:
            continue
        pred_labels[true_labels == i] = np.random.choice(
            [i, 0], np.sum(true_labels == i), p=[iou, 1 - iou]
        )

    pred_labels = relabel(pred_labels, connectivity=len(pred_labels.shape))
    return pred_labels


def simulate_predictions_accuracy(true_labels, accuracy):
    shape = true_labels.shape
    true_labels = true_labels.flatten()

    # Get the total number of labels
    n = len(true_labels)

    # Calculate the number of correct predictions
    num_correct = int(accuracy * n)

    # Create an array to store the simulated predictions (copy the true labels initially)
    simulated_predictions = np.copy(true_labels)

    # Randomly select indices to be incorrect
    incorrect_indices = np.random.choice(n, size=n - num_correct, replace=False)

    # Flip the labels at the incorrect indices
    for idx in incorrect_indices:
        # Assuming binary classification (0 or 1), flip the label
        simulated_predictions[idx] = 1 - simulated_predictions[idx]

    # Relabel the predictions
    simulated_predictions = simulated_predictions.reshape(shape)
    simulated_predictions = relabel(simulated_predictions, connectivity=len(shape))

    return simulated_predictions
