# %% Imports
from upath import UPath
import torch

from cellmap_segmentation_challenge.models import ResNet, UNet_2D

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 1  # batch size for the dataloader
input_array_info = {
    "shape": (1, 64, 64),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 64, 64),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 1  # number of epochs to train the model for
iterations_per_epoch = 3  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility
init_model_features = 8  # number of initial features for the model

classes = ["mito", "er"]  # list of classes to segment

# Defining model (comment out all that are not used)
# 2D UNet
model_name = "2d_unet"  # name of the model to use
model_to_load = "2d_unet"  # name of the pre-trained model to load
model = UNet_2D(1, len(classes))

load_model = "latest"  # load the "latest" model or the "best" validation model

# Define the paths for saving the model and logs, etc.
data_base_path = "data"  # base path where the data is stored
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# Set a limit to how long the validation can take
validation_count_limit = 1  # limit number of batches for the validation step
device = "cuda" if torch.cuda.is_available() else "cpu"  # device to use for training

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)
