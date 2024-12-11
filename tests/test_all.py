# %%
import json
import pytest
import shutil
import os

import numpy as np
from skimage.measure import label as relabel
from skimage.transform import rescale
from upath import UPath

# Set the manifest URL for the test crops
os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
)

from cellmap_segmentation_challenge import REPO_ROOT
from cellmap_segmentation_challenge.evaluate import (
    zip_submission,
    save_numpy_class_arrays_to_zarr,
)

ERROR_TOLERANCE = 1e-4


# %%
@pytest.fixture(scope="session")
def setup_temp_path(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("shared_test_dir")
    # temp_dir = (REPO_ROOT / "tests" / "tmp").absolute()
    os.environ["TEST_TMP_DIR"] = str(temp_dir)
    yield temp_dir
    # Cleanup: Unset the environment variable after tests are done
    del os.environ["TEST_TMP_DIR"]


@pytest.mark.dependency()
def test_fetch_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    os.makedirs(setup_temp_path / "data", exist_ok=True)
    fetch_data_cli.callback(
        crops="test",
        raw_padding=0,
        dest=setup_temp_path / "data",
        access_mode="append",
        fetch_all_em_resolutions=False,
        batch_size=256,
        num_workers=32,
    )


# %%
@pytest.mark.dependency()
def test_fetch_data(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    os.makedirs(setup_temp_path / "data", exist_ok=True)
    fetch_data_cli.callback(
        crops="116,234",
        raw_padding=0,
        dest=setup_temp_path / "data",
        access_mode="append",
        fetch_all_em_resolutions=False,
        batch_size=256,
        num_workers=32,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_train(setup_temp_path):
    shutil.copy(
        REPO_ROOT / "tests" / "train_config.py", setup_temp_path / "train_config.py"
    )
    # os.chdir(setup_temp_path)

    from cellmap_segmentation_challenge.cli import train_cli
    from cellmap_segmentation_challenge.utils import make_datasplit_csv

    if (setup_temp_path / "datasplit.csv").exists():
        (setup_temp_path / "datasplit.csv").unlink()

    make_datasplit_csv(
        classes=["mito", "er"],
        search_path=os.path.join(
            setup_temp_path, *"data/{dataset}/{dataset}.zarr/recon-1/{name}".split("/")
        ),
        csv_path=setup_temp_path / "datasplit.csv",
        validation_prob=0.5,
    )

    train_cli.callback(setup_temp_path / "train_config.py")


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_predict(setup_temp_path):
    from cellmap_segmentation_challenge.cli import predict_cli

    PREDICTION_PATH = os.path.join(
        setup_temp_path, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )

    predict_cli.callback(
        setup_temp_path / "train_config.py",
        crops="116",
        output_path=PREDICTION_PATH,
        do_orthoplanes=False,
        overwrite=True,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_test_crops"])
def test_predict_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import predict_cli

    PREDICTION_PATH = os.path.join(
        setup_temp_path, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )

    predict_cli.callback(
        setup_temp_path / "train_config.py",
        crops="test",
        output_path=PREDICTION_PATH,
        do_orthoplanes=False,
        overwrite=True,
    )


# %%
@pytest.mark.dependency(depends=["test_predict"])
def test_process(setup_temp_path):
    from cellmap_segmentation_challenge.cli import process_cli

    shutil.copy(
        REPO_ROOT / "tests" / "process_config.py", setup_temp_path / "process_config.py"
    )

    PREDICTION_PATH = os.path.join(
        setup_temp_path, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )
    PROCESSED_PATH = os.path.join(
        setup_temp_path, *"data/processed/{dataset}.zarr/{crop}".split("/")
    )

    process_cli.callback(
        setup_temp_path / "process_config.py",
        crops="test",
        overwrite=True,
        input_path=PREDICTION_PATH,
        output_path=PROCESSED_PATH,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_pack_results(setup_temp_path):
    from cellmap_segmentation_challenge.cli import package_submission_cli

    PROCESSED_PATH = os.path.join(
        setup_temp_path,
        *"data/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/{crop}".split("/"),
    )
    TRUTH_PATH = setup_temp_path / "data" / "truth.zarr"

    package_submission_cli.callback(PROCESSED_PATH, str(TRUTH_PATH), overwrite=True)


# %%
@pytest.mark.parametrize(
    "scale, iou, accuracy",
    [
        (None, None, None),
        (2, None, None),
        (0.5, None, None),
        (None, 0.8, 0.8),
        (2, 0.8, 0.8),
        (0.5, 0.8, 0.8),
    ],
)
@pytest.mark.dependency(depends=["test_pack_results"])
def test_evaluate(setup_temp_path, scale, iou, accuracy):
    from cellmap_segmentation_challenge.cli import evaluate_cli
    from cellmap_segmentation_challenge.evaluate import INSTANCE_CLASSES
    import zarr

    TRUTH_PATH = setup_temp_path / "data" / "truth.zarr"

    if any([scale, iou, accuracy]):
        SUBMISSION_PATH = setup_temp_path / "data" / "submission.zarr"
        submission_zarr = zarr.open(SUBMISSION_PATH, mode="w")
        truth_zarr = zarr.open(TRUTH_PATH, mode="r")
        for crop in truth_zarr.keys():
            crop_zarr = truth_zarr[crop]
            submission_zarr.create_group(crop)
            labels = []
            preds = []
            for label in crop_zarr.keys():
                label_zarr = crop_zarr[label]
                attrs = label_zarr.attrs.asdict()
                truth = label_zarr[:]

                if iou is not None and label not in INSTANCE_CLASSES:
                    pred = simulate_predictions_iou(truth, iou)
                elif accuracy is not None and label in INSTANCE_CLASSES:
                    pred = simulate_predictions_accuracy(truth, accuracy)

                if scale:
                    pred = rescale(pred, scale, order=0, preserve_range=True)
                    attrs["voxel_size"] = [s * scale for s in attrs["voxel_size"]]

                labels.append(label)
                preds.append(pred)

            save_numpy_class_arrays_to_zarr(
                SUBMISSION_PATH,
                crop,
                labels,
                preds,
                overwrite=True,
                attrs=attrs,
            )
    else:
        SUBMISSION_PATH = TRUTH_PATH
    zip_submission(SUBMISSION_PATH)

    evaluate_cli.callback(
        SUBMISSION_PATH.with_suffix(".zip"),
        result_file=setup_temp_path / "result.json",
        truth_path=TRUTH_PATH,
        instance_classes=",".join(INSTANCE_CLASSES),
    )

    # Check the results:
    with open(setup_temp_path / "result_submitted_only.json") as f:
        results = json.load(f)

    if iou is not None and accuracy is not None:
        assert (
            1 - results["overall_score"] < ERROR_TOLERANCE
        ), f"Overall score should be 1 but is: {results['overall_score']}"
    else:
        assert (
            np.abs(iou - results["overall_semantic_score"]) < ERROR_TOLERANCE
        ), f"Semantic score should be {iou} but is: {results['overall_semantic_score']}"
        # Check all accuracy scores
        for label, scores in results["label_scores"].items():
            if label in INSTANCE_CLASSES:
                assert (
                    np.abs(accuracy - scores["accuracy"]) < ERROR_TOLERANCE
                ), f"Accuracy score for {label} should be {accuracy} but is: {scores['accuracy']}"


# %%


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
