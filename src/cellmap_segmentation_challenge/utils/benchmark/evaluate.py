# %%
import numpy as np
import gunpowder as gp
from scipy.ndimage import distance_transform_edt as edt
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
