# %%
import numpy as np
import gunpowder as gp
from scipy.ndimage import distance_transform_edt as edt
from skimage.metrics import hausdorff_distance
from sklearn.metrics import jaccard_score
from pipeline import random_source_pipeline
import matplotlib.pyplot as plt

# %% First make a GT array

# SET SHAPE AND NUMBER OF POINTS HERE
shape = (256, 256, 256) # (z, y, x) shape of the volume
num_points = [shape[0] // 10, shape[0] // 5] # number of random objects in the volume [min, max]
# =====================================

src, req = random_source_pipeline(shape=shape, num_points=num_points)

with gp.build(src):
    batch = src.request_batch(req)
    array = list(batch.arrays.values())[0].data


# %%
# Plot the GT array and print some info
plt.imshow(array[shape[0]//2], cmap='tab20')
plt.colorbar()

print(f"Shape of volume: {array.shape}\nNumber of IDs: {len(np.unique(array))}")

# %%
# Euclidean distance transform on all IDs
%timeit edt(array)

# %%
# Euclidean distance transform on each ID separately
def test(array):
    for id in np.unique(array):
        this = edt(array==id)

%timeit test(array)

# %%
# Pairwise Haussdorf distance between individual labels
def h_test(array):
    ids = np.unique(array)
    for i, i1 in enumerate(ids):
        for j, i2 in enumerate(ids):
            if j<=i:
                continue
            this = hausdorff_distance(array==i1, array==i2)

%timeit h_test(array)
# %%
# Pairwise Jaccard distance between individual labels
def j_test(array):
    ids = np.unique(array)
    for i, i1 in enumerate(ids):
        for j, i2 in enumerate(ids):
            if j<=i:
                continue
            this = jaccard_score(array.flatten()==i1, array.flatten()==i2)

%timeit j_test(array)
# %%
