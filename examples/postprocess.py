# This script demonstrates how to threshold and rescale the predictions. It can be run after the predictions have been made to prepare the data for submission.

# %%
# Define the input and output paths, as well as the dataset names.
base_input_path = "./predictions_3D.zarr/{dataset_name}"
# Comment out the above line and uncomment the below line to use the 2D predictions
# base_input_path = "./predictions_2D.zarr/{dataset_name}"
dataset_names = [...]
output_path = "./{process}_predictions.zarr/{dataset_name}"

# %%
# Imports
from cellmap_segmentation_challenge import threshold_volume, rescale_volume

#
for dataset_name in dataset_names:
    dataset_path = base_input_path.format(dataset_name=dataset_name)
    output_path = output_path.format(dataset_name=dataset_name)

    # Threshold the volume
    threshold_volume(
        dataset_path,
        threshold=0.5,
        output_path=output_path.format(process="thresholded"),
    )

    # Rescale the volume
    rescale_volume(
        output_path,
        output_path.format(process="rescaled"),
        output_voxel_size=(16, 16, 16),
    )
