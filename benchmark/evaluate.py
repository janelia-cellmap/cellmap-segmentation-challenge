# %%
import numpy as np
import gunpowder as gp
from skimage.measure import label
import matplotlib.pyplot as plt

from benchmark.pipeline import (
    random_source_pipeline,
    simulate_predictions_accuracy,
    simulate_predictions_iou,
)
from cellmap_segmentation_challenge.evaluate import *

show = True

# %% First make a GT array

# SET SHAPE, NUMBER OF POINTS, ETC. HERE
size = 256  # size of the volume
shape = (size, size, size)  # (z, y, x) shape of the volume
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

print(f"Building synthetic volume with shape: {shape}")

with gp.build(src):
    batch = src.request_batch(req)
    truth_label = list(batch.arrays.values())[0].data
    truth_label = label(truth_label, connectivity=len(shape))

# %%
if show:
    # Plot the GT array and print some info
    plt.imshow(truth_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
    plt.colorbar()

print(
    f"Shape of volume: {truth_label.shape}\nNumber of IDs: {len(np.unique(truth_label))}"
)

# %%
# Make a prediction array for iou testing
pred_iou_label = simulate_predictions_iou(truth_label, configured_iou)

# Plot the prediction array
if show:
    plt.imshow(pred_iou_label[shape[0] // 2], cmap="tab20", interpolation="nearest")
    plt.colorbar()

print(
    f"Configured iou: {configured_iou}\nShape of volume: {pred_iou_label.shape}\nNumber of IDs: {len(np.unique(pred_iou_label))}"
)

# %%
print("Semantic scoring:")
semantic_score = score_semantic(pred_iou_label, truth_label)
print(semantic_score)
print(f"Configured IOU: {configured_iou}")

# # %%
# if show:
#     print("Timing semantic scoring:")
#     semantic_time = %timeit -o score_semantic(pred_iou_label, truth_label)
#     normalized_semantic_time = semantic_time.average / np.prod(shape)
#     print(f"Normalized time for semantic scoring: {normalized_semantic_time} seconds per voxel")

#     print(f"Estimate for {size}^3 volume: {normalized_semantic_time * size**3} seconds")


# %%
# Make a prediction array for accuracy testing
pred_accuracy_label = simulate_predictions_accuracy(truth_label, configured_accuracy)

# Plot the prediction array
if show:
    plt.imshow(
        pred_accuracy_label[shape[0] // 2], cmap="tab20", interpolation="nearest"
    )
    plt.colorbar()

print(
    f"Configured accuracy: {configured_accuracy}\nShape of volume: {pred_accuracy_label.shape}\nNumber of IDs: {len(np.unique(pred_accuracy_label))}"
)


# %%
print("Instance scoring:")
instance_score = score_instance(pred_accuracy_label, truth_label)
print(instance_score)
print(f"Configured accuracy: {configured_accuracy}")

# %%
# if show:
#     print("Timing instance scoring:")
#     instance_time = %timeit -o score_instance(pred_accuracy_label, truth_label)
#     normalized_instance_time = instance_time.average / np.prod(shape)
#     print(f"Normalized time for instance scoring: {normalized_instance_time} seconds per voxel")

#     print(f"Estimate for {size}^3 volume: {normalized_instance_time * size**3} seconds")
# %%

# Now let's test the whole pipeline by first saving the GT and prediction arrays to disk using the included utility functions
from cellmap_segmentation_challenge.evaluate import save_numpy_class_arrays_to_zarr

# First save the GT array
semantic_label = truth_label > 0

gt_path = "gt.zarr"
gt_array = [semantic_label, truth_label]
gt_names = ["semantic", "instance"]

save_numpy_class_arrays_to_zarr(gt_path, "test", gt_names, gt_array, overwrite=True)

# %%
# Then save the prediction arrays
pred_path = "pred.zarr"
pred_array = [pred_accuracy_label, pred_iou_label]
pred_names = ["instance", "semantic"]

save_numpy_class_arrays_to_zarr(
    pred_path, "test", pred_names, pred_array, overwrite=True
)

# Now zip it
os.system(f"zip -r {pred_path.replace('zarr', 'zip')} {pred_path}")

# %%
# Now we can score the saved arrays
from cellmap_segmentation_challenge.evaluate import score_submission

save_path = "scores.json"
submission_path = "pred.zip"
truth_path = "gt.zarr"

score_submission(
    submission_path, save_path, truth_path=truth_path, instance_classes=["instance"]
)
# %%
