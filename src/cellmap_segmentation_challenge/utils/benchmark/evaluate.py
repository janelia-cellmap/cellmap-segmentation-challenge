# %%
import numpy as np
import gunpowder as gp
from scipy.optimize import linear_sum_assignment
from skimage.metrics import hausdorff_distance
from sklearn.metrics import jaccard_score
from skimage.measure import label as relabel
from pipeline import random_source_pipeline
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
    array = list(batch.arrays.values())[0].data

# %%
# Plot the GT array and print some info
plt.imshow(array[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Shape of volume: {array.shape}\nNumber of IDs: {len(np.unique(array))}")

# %%
# Make a prediction array
ratio = 0.7  # ratio of correct pixels to incorrect pixels
pred = np.zeros_like(array)
truth_mask = np.random.rand(*array.shape) < ratio
pred[truth_mask] = array[truth_mask]
false_positives = (truth_mask == 0) * (array == 0)
pred[false_positives] = 1
pred = relabel(pred)

# Plot the prediction array
plt.imshow(pred[shape[0] // 2], cmap="tab20", interpolation="nearest")
plt.colorbar()

print(f"Shape of volume: {pred.shape}\nNumber of IDs: {len(np.unique(pred))}")

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
from cellmap_segmentation_challenge.utils.evaluate import score_instance, score_semantic

# %%
print("Instance scoring:")
instance_score = score_instance(pred, array)
print(instance_score)
print(f"Configured accuracy: {ratio}")
# %%

# %%
print("Timing instance scoring:")
instance_time = %timeit -o score_instance(pred, array)
normalized_instance_time = instance_time.average / np.prod(shape)
print(f"Normalized time for instance scoring: {normalized_instance_time} seconds per voxel")

size = 512
print(f"Estimate for {size}^3 volume: {normalized_instance_time * size**3} seconds")
# %%
print("Semantic scoring:")
semantic_score = score_semantic(pred, array)
print(semantic_score)
# %%
print("Timing semantic scoring:")
semantic_time = %timeit -o score_semantic(pred, array)
normalized_semantic_time = semantic_time.average / np.prod(shape)
print(f"Normalized time for semantic scoring: {normalized_semantic_time} seconds per voxel")

size = 512
print(f"Estimate for {size}^3 volume: {normalized_semantic_time * size**3} seconds")

# %%
