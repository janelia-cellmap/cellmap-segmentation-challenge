# %% Imports
import os
import torch
import numpy as np
from tqdm import tqdm, trange
from utils import get_dataloader, save_result_figs, get_loss_plot, CellMapLossWrapper
from models import unet_model, ResNet
from tensorboardX import SummaryWriter
from cellmap_data.utils import get_image_dict

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 8  # batch size for the dataloader
input_array_info = {
    "shape": (1, 128, 128),
    "scale": (64, 64, 64),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (1, 128, 128),
    "scale": (64, 64, 64),
}  # shape and voxel size of the data to load for the target
epochs = 10  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility
init_model_features = 32  # number of initial features for the model

classes = ["nuc"]  # list of classes to segment
model_name = "2d_resnet"  # name of the model to use
data_base_path = "data"  # base path where the data is stored
logs_save_path = "tensorboard/{model_name}"  # path to save the logs from tensorboard
model_save_path = (
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
)
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use

# %% Make sure the save path exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# %% Set the random seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# %% Check that the GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

# %% Download the data and make the dataloader
train_loader, val_loader = get_dataloader(
    datasplit_path,
    classes,
    batch_size=batch_size,
    input_array_info=input_array_info,
    target_array_info=target_array_info,
    iterations_per_epoch=iterations_per_epoch,
    device=device,
)

# %% Define the model and move model to device
model = model = ResNet(ndims=2, input_nc=1, output_nc=len(classes))
model = model.to(device)

# %% Define the optimizer
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

# %% Define the loss function
criterion = torch.nn.BCEWithLogitsLoss

# Use custom loss function wrapper that handles NaN values in the target. This works with any PyTorch loss function
criterion = CellMapLossWrapper(criterion)

# %% Train the model
post_fix_dict = {}

# Define a summarywriter
writer = SummaryWriter(logs_save_path.format(model_name=model_name))

# Create a variable to track iterations
n_iter = 0
# Training outer loop, across epochs
for epoch in range(epochs):

    # Set the model to training mode to enable backpropagation
    model.train()

    # Training loop for the epoch
    post_fix_dict["Epoch"] = epoch + 1
    epoch_bar = tqdm(train_loader.loader, desc="Training")
    for batch in epoch_bar:
        # Increment the training iteration
        n_iter += 1

        # Get the inputs and targets
        inputs = batch["input"]
        targets = batch["output"]

        # Zero the gradients, so that they don't accumulate across iterations
        optimizer.zero_grad()

        # Forward pass (compute the output of the model)
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass (compute the gradients)
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the progress bar
        post_fix_dict["Loss"] = f"{loss.item()}"
        epoch_bar.set_postfix(post_fix_dict)

        # Log the loss using tensorboard
        writer.add_scalar("loss", loss.item(), n_iter)

    # Save the model
    torch.save(
        model.state_dict(),
        model_save_path.format(epoch=epoch + 1, model_name=model_name),
    )

    # Set the model to evaluation mode to disable backpropagation
    model.eval()

    # Compute the validation score by averaging the loss across the validation set
    val_score = 0
    val_bar = tqdm(val_loader, desc="Validation")
    for batch in val_bar:

        inputs = batch["input"]
        targets = batch["output"]
        outputs = model(inputs)
        val_score += criterion(outputs, targets).item()

    val_score /= len(val_loader)
    # Log the validation using tensorboard
    writer.add_scalar("validation", val_score, n_iter)

    # Update the progress bar
    post_fix_dict["Validation"] = f"{val_score:.4f}"

    # Generate and save figures from the last batch of the validation to appear in tensorboard
    figs = get_image_dict(inputs, outputs, targets, classes)
    for name, fig in figs.items():
        writer.add_figure(name, fig, n_iter)

    # Refresh the train loader to shuffle the data yielded by the dataloader
    train_loader.refresh()

# Close the summarywriter
writer.close()

# %%
