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
import numpy as np
import gunpowder as gp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import dice, jaccard
from skimage.metrics import hausdorff_distance
from sklearn.metrics import jaccard_score
from skimage.measure import label
import matplotlib.pyplot as plt

from cellmap_segmentation_challenge.utils.benchmark.pipeline import random_source_pipeline, simulate_predictions_accuracy, simulate_predictions_iou
from cellmap_segmentation_challenge.utils.evaluate import *


# %% First make a GT array

# SET SHAPE, NUMBER OF POINTS, ETC. HERE
shape = (128, 128, 128)  # (z, y, x) shape of the volume
num_points = [
    shape[0] // 10,
    shape[0] // 5,
]  # number of random objects in the volume [min, max]

configured_accuracy = 0.8
configured_iou = 0.8
# =====================================

src, req = random_source_pipeline(
    shape=shape, num_points=num_points, relabel_connectivity=len(shape)
)

with gp.build(src):
    batch = src.request_batch(req)
    truth_label = list(batch.arrays.values())[0].data
    truth_label = label(truth_label, connectivity=len(shape))

# %%
# Plot the GT array and print some info
plt.imshow(truth_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Shape of volume: {truth_label.shape}\nNumber of IDs: {len(np.unique(truth_label))}")

# %%
# Make a prediction array for iou testing
pred_iou_label = simulate_predictions_iou(truth_label, configured_iou)

# Plot the prediction array
plt.imshow(pred_iou_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Configured iou: {configured_iou}\nShape of volume: {pred_iou_label.shape}\nNumber of IDs: {len(np.unique(pred_iou_label))}")

#%%
print("Semantic scoring:")
semantic_score = score_semantic(pred_iou_label, truth_label)
print(semantic_score)
print(f"Configured IOU: {configured_iou}")

# %%
print("Timing semantic scoring:")
semantic_time = %timeit -o score_semantic(pred_iou_label, truth_label)
normalized_semantic_time = semantic_time.average / np.prod(shape)
print(f"Normalized time for semantic scoring: {normalized_semantic_time} seconds per voxel")

size = 512
print(f"Estimate for {size}^3 volume: {normalized_semantic_time * size**3} seconds")


# %%
# Make a prediction array for accuracy testing
pred_accuracy_label = simulate_predictions_accuracy(truth_label, configured_accuracy)

# Plot the prediction array
plt.imshow(pred_accuracy_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Configured accuracy: {configured_accuracy}\nShape of volume: {pred_accuracy_label.shape}\nNumber of IDs: {len(np.unique(pred_accuracy_label))}")


# %%
print("Instance scoring:")
instance_score = score_instance(pred_accuracy_label, truth_label)
print(instance_score)
print(f"Configured accuracy: {configured_accuracy}")

# %%
print("Timing instance scoring:")
instance_time = %timeit -o score_instance(pred_accuracy_label, truth_label)
normalized_instance_time = instance_time.average / np.prod(shape)
print(f"Normalized time for instance scoring: {normalized_instance_time} seconds per voxel")

size = 512
print(f"Estimate for {size}^3 volume: {normalized_instance_time * size**3} seconds")
# %%

# Now let's test the whole pipeline by first saving the GT and prediction arrays to disk using the included utility functions
from cellmap_segmentation_challenge.utils.evaluate import save_numpy_class_arrays_to_zarr
# First save the GT array

