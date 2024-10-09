# %%
from cellmap_segmentation_challenge.models import UNet_3D, ResNet, ViTVNet

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 6  # batch size for the dataloader
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 1000  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility
init_model_features = 32  # number of initial features for the model

classes = ["nuc", "er"]  # list of classes to segment


# Defining model (comment out all that are not used)
# 3D UNet
model_name = "3d_unet"  # name of the model to use
model_to_load = "3d_unet"  # name of the pre-trained model to load
model = UNet_3D(1, len(classes))

# 3D ResNet
# model_name = "3d_resnet"  # name of the model to use
# model_to_load = "3d_resnet"  # name of the pre-trained model to load
# model = ResNet(ndims=3, output_nc=len(classes))

# # 3D ViT VNet
# model_name = "3d_vnet"  # name of the model to use
# model_to_load = "3d_vnet"  # name of the pre-trained model to load
# model = ViTVNet(len(classes))

load_model = "latest"  # load the latest model or the best validation model

# Define the paths for saving the model and logs, etc.
data_base_path = "data"  # base path where the data is stored
logs_save_path = "tensorboard/{model_name}"  # path to save the logs from tensorboard
model_save_path = (
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
)
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y", "z"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    train(__file__)
