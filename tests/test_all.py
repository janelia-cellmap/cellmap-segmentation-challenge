# %%
import json
import pytest
import shutil
import os

import numpy as np
from skimage.measure import label as relabel
from skimage.transform import rescale
import requests

# Set the manifest URL for the test crops
os.environ["CSC_TEST_CROP_MANIFEST_URL"] = (
    "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/test_crop_manifest.csv"
)

from cellmap_segmentation_challenge import REPO_ROOT, RAW_NAME, CROP_NAME
from cellmap_segmentation_challenge.evaluate import (
    zip_submission,
    save_numpy_class_arrays_to_zarr,
)

ERROR_TOLERANCE = 0.1


# %%
@pytest.fixture(scope="session")
def setup_temp_path(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("shared_test_dir")
    # temp_dir = (REPO_ROOT / "tests" / "tmp").absolute()  # For debugging
    yield temp_dir


@pytest.mark.dependency()
def test_fetch_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import fetch_data_cli

    os.makedirs(setup_temp_path / "data", exist_ok=True)
    fetch_data_cli.callback(
        crops="test",
        raw_padding=2,
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
        crops="116,118",
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
    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/train_config.py",
        setup_temp_path / "train_config.py",
    )

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

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/train_config.py",
        setup_temp_path / "train_config.py",
    )

    prediction_path = os.path.join(
        setup_temp_path, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )
    search_path = os.path.join(
        setup_temp_path, *"data/{dataset}/{dataset}.zarr/recon-1/{name}".split("/")
    )

    predict_cli.callback(
        setup_temp_path / "train_config.py",
        crops="116",
        output_path=prediction_path,
        do_orthoplanes=False,
        overwrite=True,
        search_path=search_path,
        raw_name=RAW_NAME,
        crop_name=CROP_NAME,
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_test_crops"])
def test_predict_test_crops(setup_temp_path):
    from cellmap_segmentation_challenge.cli import predict_cli

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/train_config.py",
        setup_temp_path / "train_config.py",
    )

    prediction_path = os.path.join(
        setup_temp_path, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )
    search_path = os.path.join(
        setup_temp_path, *"data/{dataset}/{dataset}.zarr/recon-1/{name}".split("/")
    )

    predict_cli.callback(
        setup_temp_path / "train_config.py",
        crops="test",
        output_path=prediction_path,
        do_orthoplanes=False,
        overwrite=True,
        search_path=search_path,
        raw_name=RAW_NAME,
        crop_name=CROP_NAME,
    )


# %%
@pytest.mark.dependency(depends=["test_predict"])
def test_process(setup_temp_path):
    from cellmap_segmentation_challenge.cli import process_cli

    download_file(
        "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/tests/process_config.py",
        setup_temp_path / "process_config.py",
    )

    prediction_path = os.path.join(
        setup_temp_path, *"data/predictions/{dataset}.zarr/{crop}".split("/")
    )
    processed_path = os.path.join(
        setup_temp_path, *"data/processed/{dataset}.zarr/{crop}".split("/")
    )

    process_cli.callback(
        setup_temp_path / "process_config.py",
        crops="test",
        overwrite=True,
        input_path=prediction_path,
        output_path=processed_path,
        device="cpu",
    )


# %%
@pytest.mark.dependency(depends=["test_fetch_data"])
def test_pack_results(setup_temp_path):
    from cellmap_segmentation_challenge.cli import package_submission_cli

    processed_path = os.path.join(
        setup_temp_path,
        *"data/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/{crop}".split("/"),
    )
    truth_path = setup_temp_path / "data" / "truth.zarr"

    package_submission_cli.callback(processed_path, str(truth_path), overwrite=True)


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

    truth_path = setup_temp_path / "data" / "truth.zarr"

    if any([scale, iou, accuracy]):
        submission_path = setup_temp_path / "data" / "submission.zarr"
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
                    pred = simulate_predictions_iou(pred, iou)
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
        result_file=setup_temp_path / "result.json",
        truth_path=truth_path,
        instance_classes=",".join(INSTANCE_CLASSES),
    )

    # Check the results:
    with open(setup_temp_path / "result_submitted_only.json") as f:
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


# %%


# Helper functions for simulating predictions
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


def download_file(url, dest):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)
