# %%
import re
import numpy as np
from scipy import ndimage
import tensorstore as ts
from upath import UPath
import h5py

from shared import TRUTH_DATASETS, RESOLUTION_LEVELS, CLASS_DATASETS

import logging

logger = logging.getLogger(__name__)


METRICS = [
    "false_positives",
    "false_negatives",
    "false_positive_stats",
    "false_negative_stats",
    "iou",
    "class_score",
]

SUMMARY_METRICS = ["false_positives", "false_negatives", "iou", "class_score"]


def _zarr_spec_from_path(path: str) -> ...:
    if re.match(r"\w+\://", path):  # path is a URI
        kv_store = path
    else:
        kv_store = {"driver": "file", "path": path}
    return {"driver": "zarr", "kvstore": kv_store}


def load_data(path: str):
    assert UPath(path).exists()
    if ".zarr" in path:
        spec = _zarr_spec_from_path(path)
        return np.array(ts.open(spec, read=True, write=False).result())
    elif ".h5" in path or ".hdf5" in path:
        with h5py.File(path, "r") as f:
            return f["data"][:]


class ClassError:
    """
    Calculates the error of a single class in a segmentation of a single volume. This includes the following metrics:
    - False positives
    - False negatives
    - Intersection over Union (IoU)
    - False positive distance statistics (mean, std, max, count, median)
    - False negative distance statistics (mean, std, max, count, median)
    Functions were adapted from https://github.com/cremi/cremi_python
    This class is used by the TestScore class to calculate the error across all test data.
    """

    def __init__(
        self,
        resolution: tuple,
        test: np.ndarray,
        truth: np.ndarray,
        distance_max: float = 80.0,
        norm_slope: float = 1.0,
    ):
        """
        Parameters
        ----------
        resolution : tuple
            The resolution of the data in nanometers per pixel.
        test : np.ndarray
            The predicted segmentation.
        truth : np.ndarray
            The ground truth segmentation.
        distance_max : float, optional
            The maximum distance to consider for distance-based metrics, by default 80.
        norm_slope : float, optional
            The slope of the tanh function used to normalize distances, by default 1.
        """

        self.resolution = resolution
        self.distance_max = distance_max
        self.norm_slope = norm_slope
        self.test = test
        self.truth = truth

        if self.test.shape != self.truth.shape:
            # Crop the test data to the shape of the ground truth data
            truth_shape = np.array(truth.shape)
            test_shape = np.array(test.shape)
            if any(test_shape > truth_shape):
                logger.warning(
                    "Test data is larger than the ground truth data. Cropping to the shape of the ground truth data - assumes the ground truth data is centered in the test data."
                )
                crop = (test_shape - truth_shape) // 2
                self.test = self.test[
                    crop[0] : crop[0] + truth.shape[0],
                    crop[1] : crop[1] + truth.shape[1],
                    crop[2] : crop[2] + truth.shape[2],
                ]
            else:
                raise ValueError(
                    "Test data must be the same shape as the ground truth data, or larger."
                )

        self.test_edt = self.normalize_distances(
            ndimage.distance_transform_edt(self.test, sampling=self.resolution)
        )
        self.truth_edt = self.normalize_distances(
            ndimage.distance_transform_edt(self.truth, sampling=self.resolution)
        )

    def normalize_distances(self, x):
        # Normalized smoothly to between 0 and max_distance
        # return self.distance_max * np.tanh(self.norm_slope * (x / self.distance_max))

        # Normalized smoothly to between 0 and 1
        return np.tanh(self.norm_slope * (x / self.distance_max))

        # Clipped to between 0 and 1, scaled to distance_max
        # return np.clip(x, 0, self.distance_max) / self.distance_max

    @property
    def test_background(self):
        if hasattr(self, "_test_background"):
            return self._test_background
        else:
            self._test_background = self.test == 0
        return self.test_background

    @property
    def truth_background(self):
        if hasattr(self, "_truth_background"):
            return self._truth_background
        else:
            self._truth_background = self.truth == 0
        return self.truth_background

    def false_positives(self, threshold=40):
        mask2 = self.truth_edt > threshold
        false_positives = self.truth_edt[np.logical_and(self.test_background, mask2)]

        return false_positives.size

    def false_negatives(self, threshold=40):
        mask2 = self.test_edt > threshold
        false_negatives = self.test_edt[np.logical_and(self.truth_background, mask2)]

        return false_negatives.size

    def false_positive_stats(self):
        if hasattr(self, "_false_positive_stats"):
            return self._false_positive_stats
        else:
            false_positives = self.truth_edt[self.test_background]
            self._false_positive_stats = {
                "mean": np.mean(false_positives),
                "std": np.std(false_positives),
                "max": np.amax(false_positives),
                "count": false_positives.size,
                "median": np.median(false_positives),
            }
        return self._false_positive_stats

    def false_negative_stats(self):
        if hasattr(self, "_false_negative_stats"):
            return self._false_negative_stats
        else:
            false_negatives = self.test_edt[self.truth_background]
            self._false_negative_stats = {
                "mean": np.mean(false_negatives),
                "std": np.std(false_negatives),
                "max": np.amax(false_negatives),
                "count": false_negatives.size,
                "median": np.median(false_negatives),
            }
        return self._false_negative_stats

    def iou(self):
        if hasattr(self, "_iou"):
            return self._iou
        else:
            intersection = np.logical_and(self.test, self.truth)
            union = np.logical_or(self.test, self.truth)
            self._iou = np.sum(intersection) / np.sum(union)
        return self._iou

    def class_score(self):
        """
        This function returns a summary score for each class per dataset, for easy agglomeration across datasets and classes.
        """
        if hasattr(self, "_class_score"):
            return self._class_score
        else:
            self._class_score = self.iou()
        return self._class_score

    def return_results(self, metrics=METRICS):
        return {k: getattr(self, k)() for k in metrics}


# %%
def score(
    label_dict: dict[str, int],
    test_datasets: dict[str, str],
    resolution: int,
    distance_max: float = 80.0,
    norm_slope: float = 1.0,
    truth_datasets: dict[str, str] = TRUTH_DATASETS,
):
    """
    Parameters
    ----------
    label_dict : dict[str, int]
        A dictionary mapping class labels to their index in the test_datasets segmentation.
    test_datasets : dict[str, str]
        A dictionary mapping dataset names to their paths.
    resolution : int
        The resolution of the data in nanometers per pixel. This should be one of the following: 8, 16, 32, 64, 128, 256, 512, 1024, 2048. Data should be isotropic.
    distance_max : float, optional
        The maximum distance to consider for distance-based metrics, by default 80.
    norm_slope : float, optional
        The slope of the tanh function used to normalize distances, by default 1.
    truth_datasets : dict[str, str], optional
        A dictionary mapping dataset names to their ground truth paths, by default TRUTH_DATASETS. Dataset names must match those in test_datasets, and are expected to have the format: `provided_path/{label}/{resolution_level}`.

    Returns
    -------
    float
        The overall score across all classes and datasets.
    dict
        A dictionary mapping class labels to dictionaries of summary metrics.
    dict
        A dictionary mapping dataset names to dictionaries of class metrics.
    """
    resolution_level = RESOLUTION_LEVELS[resolution]
    dataset_results = {k: {} for k in test_datasets.keys()}
    for dataset, path in test_datasets.items():
        try:
            test = load_data(path)
        except:
            logger.error(f"Could not load test data for {dataset}")
            continue

        for label, idx in label_dict.items():
            try:
                truth = load_data(
                    truth_datasets[dataset].format(
                        label=label, resolution_level=resolution_level
                    )
                )
            except:
                logger.error(
                    f"Could not load groundtruth data for {label} in {dataset}"
                )
                continue
            if len(label_dict) == 1:
                class_test = test.squeeze()
            else:
                # If there are multiple classes, extract the class of interest
                class_test = test[idx]

            class_error = ClassError(
                (resolution,) * 3, class_test, truth, distance_max, norm_slope
            )
            dataset_results[dataset][label] = class_error.return_results()

    summary_results = {k: {} for k in label_dict.keys()}
    overall_score = []
    for label in label_dict.keys():
        for metric in SUMMARY_METRICS:
            summary_results[label][metric] = np.mean(
                [
                    dataset_results[dataset][label][metric]
                    for dataset in test_datasets.keys()
                ]
            )
        overall_score.append(summary_results[label]["class_score"])

    overall_score = np.mean(overall_score)

    return overall_score, summary_results, dataset_results


def get_leaderboard_stat(label: str, dataset_results):
    """
    This function calculates the leaderboard statistic for a given class across all leaderboard datasets.

    Parameters
    ----------
    label : str
        The class label to calculate the leaderboard statistic for.
    dataset_results : dict
        A dictionary of dataset results as returned by the score function.

    Returns
    -------
    float
        The mean class score across all leaderboard datasets.
    float
        The standard deviation of the class score across all leaderboard datasets.

    Raises
    ------
    ValueError
        If a leaderboard dataset is missing from the dataset_results.
    """
    leaderboard_stats = ()
    for dataset in CLASS_DATASETS[label]:
        if dataset in dataset_results:
            leaderboard_stats += (dataset_results[dataset][label]["class_score"],)
        else:
            raise ValueError(
                f"Dataset {dataset} not found in dataset_results for class {label}. This is not a valid submission."
            )
    return np.mean(leaderboard_stats), np.std(leaderboard_stats)


def get_leaderboard_stats(
    label_dict: dict[str, int],
    test_datasets: dict[str, str],
    resolution: int,
    distance_max: float = 80.0,
    norm_slope: float = 1.0,
    truth_datasets: dict[str, str] = TRUTH_DATASETS,
):
    """
    This function calculates the leaderboard statistics for each class across all leaderboard datasets at a given resolution.

    Parameters
    ----------
    label_dict : dict[str, int]
        A dictionary mapping class labels to their index in the test_datasets segmentation.
    test_datasets : dict[str, str]
        A dictionary mapping dataset names to their paths.
    resolution : int
        The resolution of the data in nanometers per pixel. This should be one of the following: 8, 16, 32, 64, 128, 256, 512, 1024, 2048. Data should be isotropic.
    distance_max : float, optional
        The maximum distance to consider for distance-based metrics, by default 80.
    norm_slope : float, optional
        The slope of the tanh function used to normalize distances, by default 1.
    truth_datasets : dict[str, str], optional
        A dictionary mapping dataset names to their ground truth paths, by default TRUTH_DATASETS. Dataset names must match those in test_datasets, and are expected to have the format: `provided_path/{label}/{resolution_level}`.
    """

    _, _, dataset_results = score(
        label_dict, test_datasets, resolution, distance_max, norm_slope, truth_datasets
    )

    leaderboard_stats = {k: {} for k in label_dict.keys()}
    for label in label_dict.keys():
        leaderboard_stats[label]["mean"], leaderboard_stats[label]["std"] = (
            get_leaderboard_stat(label, dataset_results)
        )

    return leaderboard_stats


def print_leaderboard_stats(leaderboard_stats):
    for label, stats in leaderboard_stats.items():
        print(f"{label}: {stats['mean']} +/- {stats['std']}")
    print(f"Overall: {np.mean([v['mean'] for v in leaderboard_stats.values()])}")


# %%
# Example usage
import matplotlib.pyplot as plt

label_dict = {"mito": 0}
test_datasets = {
    "jrc_hela-2": "/nrs/cellmap/bennettd/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155/mito/s0"
}
resolution = 8
distance_max = 80.0
norm_slope = 1.0
truth_datasets = {
    "jrc_hela-2": "/nrs/cellmap/bennettd/data/jrc_hela-2/jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155/{label}/{resolution_level}"
}
overall_score, summary_results, dataset_results = score(
    label_dict, test_datasets, resolution, distance_max, norm_slope, truth_datasets
)

# %%
# leaderboard_stats = get_leaderboard_stats(
#     label_dict, test_datasets, resolution, distance_max, norm_slope, truth_datasets
# )
CLASS_DATASETS = {"mito": ["jrc_hela-2"]}
leaderboard_stats = {k: {} for k in label_dict.keys()}
for label in label_dict.keys():
    leaderboard_stats[label]["mean"], leaderboard_stats[label]["std"] = (
        get_leaderboard_stat(label, dataset_results)
    )
print_leaderboard_stats(leaderboard_stats)

# %%
