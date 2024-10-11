# This file is used to predict the segmentation logits of the 3D dataset using the model trained in the train_2D.py script.
# %%
from cellmap_segmentation_challenge.utils.datasplit import RAW_NAME
from cellmap_segmentation_challenge import predict_ortho_planes

# Define the input and output paths, as well as the dataset names.
base_input_path = f"./test.zarr/{{dataset_name}}/{RAW_NAME}"
dataset_names = [...]  # Define the dataset names
base_output_path = "./predictions_2D.zarr/{dataset_name}"

# Load the model specified in the train_3D.py script (will load the latest model or the one that performed best on the validation set, depending on the script)
from train_2D import model, classes, input_array_info

for dataset_name in dataset_names:
    dataset_path = base_input_path.format(dataset_name=dataset_name)
    output_path = base_output_path.format(dataset_name=dataset_name)

    # Predict the segmentation of the dataset
    predict_ortho_planes(
        model, dataset_path, output_path, input_array_info["shape"], classes
    )
