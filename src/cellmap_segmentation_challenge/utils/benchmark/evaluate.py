# %%
import numpy as np
import gunpowder as gp
from scipy.optimize import linear_sum_assignment
from skimage.metrics import hausdorff_distance
from sklearn.metrics import jaccard_score
from skimage.measure import label as relabel
from cellmap_segmentation_challenge.utils.benchmark.pipeline import random_source_pipeline
import matplotlib.pyplot as plt

# %% First make a GT array

# SET SHAPE AND NUMBER OF POINTS HERE
shape = (128, 128, 128)  # (z, y, x) shape of the volume
num_points = [
    shape[0] // 10,
    shape[0] // 5,
]  # number of random objects in the volume [min, max]
# =====================================

src, req = random_source_pipeline(
    shape=shape, num_points=num_points, relabel_connectivity=len(shape)
)

with gp.build(src):
    batch = src.request_batch(req)
    truth_label = list(batch.arrays.values())[0].data

# %%
# Plot the GT array and print some info
plt.imshow(truth_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Shape of volume: {truth_label.shape}\nNumber of IDs: {len(np.unique(truth_label))}")

# %%
# Make a prediction array
false_negative_rate = 0.3
false_positive_rate = 0.1

pred_label = truth_label.copy()

actual_positives = np.sum(truth_label > 0)
false_negatives = np.random.choice([0, 1], actual_positives, p=[false_negative_rate, 1-false_negative_rate])
pred_label[truth_label > 0] = false_negatives * truth_label[truth_label > 0]

# truth_mask = np.random.rand(*truth_label.shape) > false_negative_rate
# pred_label[truth_mask] = truth_label[truth_mask]

actual_negatives = np.sum(truth_label == 0)
false_positives = np.random.choice([1, 0], actual_negatives, p=[false_positive_rate, 1-false_positive_rate])
pred_label[truth_label == 0] = false_positives

# false_positives = np.random.rand(*truth_label.shape) < false_positive_rate
# false_positives = np.logical_and(false_positives, truth_label == 0)
# pred_label[false_positives] = np.random.choice(
#     [0,1], np.sum(false_positives), p=[1-false_positive_rate, false_positive_rate]
# )

def simulate_predictions(true_labels, accuracy):
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
    
    return simulated_predictions

configured_accuracy = ((1-false_negative_rate) + (1-false_positive_rate))/2

pred_label = simulate_predictions(truth_label.flatten(), configured_accuracy).reshape(truth_label.shape)

pred_label = relabel(pred_label, connectivity=len(shape))

# Plot the prediction array
plt.imshow(pred_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Shape of volume: {pred_label.shape}\nNumber of IDs: {len(np.unique(pred_label))}")

# %%
"""
Add tolerance based on known GT precision --> mask out uncertain pixels (distance transform then threshold)
Upsample GT to match prediction resolution --> should be native resolution (8nm/)
Determine tolerance per label --> Aubrey
Combined metric:
- geometric mean of Jaccard score and Hausdorff distance

For semantic:
- Jaccard score
- Hausdorff distance w/ cutoff or sigmoid

For instance:
- Jaccard score
--> Match by best Jaccard score
    - False positive = unmatched pred
    - False negative = unmatched GT
    - Count false splits
        - near matches over a threshold are incorrect splits where the GT is split into multiple predictions
    - Count false merges
        - near matches over a threshold are incorrect merges where multiple GT are merged into a single prediction
- Hausdorff distance w/ cutoff or sigmoid

"""
# %%
from cellmap_segmentation_challenge.utils.evaluate import *

# %%
print("Instance scoring:")
instance_score = score_instance(pred_label, truth_label)
print(instance_score)
configured_accuracy = ((1-false_negative_rate) + (1-false_positive_rate))/2
print(f"Configured accuracy: {configured_accuracy}")

# %%
print("Timing instance scoring:")
instance_time = %timeit -o score_instance(pred_label, truth_label)
normalized_instance_time = instance_time.average / np.prod(shape)
print(f"Normalized time for instance scoring: {normalized_instance_time} seconds per voxel")

size = 512
print(f"Estimate for {size}^3 volume: {normalized_instance_time * size**3} seconds")
# %%
print("Semantic scoring:")
semantic_score = score_semantic(pred_label, truth_label)
print(semantic_score)

# %%
print("Timing semantic scoring:")
semantic_time = %timeit -o score_semantic(pred_label, truth_label)
normalized_semantic_time = semantic_time.average / np.prod(shape)
print(f"Normalized time for semantic scoring: {normalized_semantic_time} seconds per voxel")

size = 512
print(f"Estimate for {size}^3 volume: {normalized_semantic_time * size**3} seconds")

# %%
