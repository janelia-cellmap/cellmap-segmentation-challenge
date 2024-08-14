# %%
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class ClassError:
    """
    Calculates the error of a single class in a segmentation of a single volume. This includes the following metrics:
    - False positives
    - False negatives
    - Intersection over Union (IoU)
    - False positive distance statistics (mean, std, max, count, median)
    - False negative distance statistics (mean, std, max, count, median)
    This class is used by the TestScore class to calculate the error across all test data.
    """

    def __init__(self, resolution, test, truth, distance_max=80, norm_slope=1):
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

    def count_false_positives(self, threshold=40):

        mask1 = self.test == 0
        mask2 = self.truth_edt > threshold
        false_positives = self.truth_edt[np.logical_and(mask1, mask2)]

        return false_positives.size

    def count_false_negatives(self, threshold=40):

        mask1 = self.truth == 0
        mask2 = self.test_edt > threshold
        false_negatives = self.test_edt[np.logical_and(mask1, mask2)]

        return false_negatives.size

    def acc_false_positives(self):

        mask = self.test == 0
        false_positives = self.truth_edt[mask]
        stats = {
            "mean": np.mean(false_positives),
            "std": np.std(false_positives),
            "max": np.amax(false_positives),
            "count": false_positives.size,
            "median": np.median(false_positives),
        }
        return stats

    def acc_false_negatives(self):

        mask = self.truth == 0
        false_negatives = self.test_edt[mask]
        stats = {
            "mean": np.mean(false_negatives),
            "std": np.std(false_negatives),
            "max": np.amax(false_negatives),
            "count": false_negatives.size,
            "median": np.median(false_negatives),
        }
        return stats

    def iou(self):

        intersection = np.logical_and(self.test, self.truth)
        union = np.logical_or(self.test, self.truth)

        iou = np.sum(intersection) / np.sum(union)

        return iou


# %%
class TestScore:
    def __init__(
        self, labels, resolution, test_datasets, distance_max=80, norm_slope=1
    ): ...
