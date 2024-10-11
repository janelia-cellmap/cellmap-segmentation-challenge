# This is an example of a training configuration file that trains a 2D U-Net model to predict nuclei and endoplasmic reticulum in the CellMap Segmentation Challenge dataset.

# The configuration file defines the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file as an argument to start the training process. The `train` function reads the configuration file, sets up the data loaders, model, optimizer, loss function, and other components, and trains the model for the specified number of epochs.

# The configuration file includes the following components:
# 1. Hyperparameters: learning rate, batch size, input and target array information, epochs, iterations per epoch, random seed, and initial number of features for the model.
# 2. Model: 2D U-Net model with two classes (nuclei and endoplasmic reticulum). (You can also use a 2D ResNet model by uncommenting the relevant lines.)
# 3. Paths: paths for saving logs, model checkpoints, and data split file.
# 4. Spatial transformations: spatial transformations to apply to the training data.

# This configuration file can be used to run training via two different commands:
# 1. `python train_2D.py`: Run the training script directly.
# 2. `csc train train_2D.py`: Run the training script using the `csc train` command-line interface.

# Training progress can be monitored using TensorBoard by running `tensorboard --logdir tensorboard` in the terminal.

# Once the model is trained, you can use the `predict` function to make predictions on new data using the trained model. See the `predict_2D.py` example for more details.

# %% Imports
from cellmap_segmentation_challenge.models import UNet_2D, ResNet

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 8  # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 1000  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility
init_model_features = 32  # number of initial features for the model

classes = ["nuc", "er"]  # list of classes to segment

# Defining model (comment out all that are not used)
# 2D UNet
model_name = "2d_unet"  # name of the model to use
model_to_load = "2d_unet"  # name of the pre-trained model to load
model = UNet_2D(1, len(classes))

# # 2D ResNet [uncomment to use]
# model_name = "2d_resnet"  # name of the model to use
# model_to_load = "2d_resnet"  # name of the pre-trained model to load
# model = ResNet(ndims=2, output_nc=len(classes))

load_model = "latest"  # load the "latest" model or the "best" validation model

# Define the paths for saving the model and logs, etc.
data_base_path = "data"  # base path where the data is stored
logs_save_path = "tensorboard/{model_name}"  # path to save the logs from tensorboard
model_save_path = (
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
)
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    # Call the train function with the configuration file
    train(__file__)
