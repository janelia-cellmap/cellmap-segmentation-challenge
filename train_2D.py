# %% Imports
import os
import torch
import numpy as np
from tqdm import tqdm, trange
from utils.dataloader import get_dataloader
from utils.visualize import save_result_figs, get_loss_plot
from utils.loss import CellMapLossWrapper

# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 32  # batch size for the dataloader
array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load
epochs = 10  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
num_workers = 4  # number of workers for the dataloader to use
random_seed = 42  # random seed for reproducibility
init_model_features = 32  # number of initial features for the model

classes = ["nuc"]  # list of classes to segment
model_name = "2D_unet"  # name of the model to use
data_base_path = "data"  # base path where the data is stored
figures_save_path = (
    "figures/{model_name}/{epoch}/{label}.png"  # path to save the example figures
)
model_save_path = (
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
)
datasplit_path = "datasplit.csv"  # path to the datasplit file that defines the train/val split the dataloader should use


# %% Make sure the save path exists
for path in [figures_save_path, model_save_path]:
    os.makedirs(os.path.dirname(figures_save_path), exist_ok=True)

# %% Set the random seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# %% Check that the GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %% Download the data and make the dataloader
train_loader, val_loader = get_dataloader(
    datasplit_path,
    classes,
    batch_size=batch_size,
    array_info=array_info,
    iterations_per_epoch=iterations_per_epoch,
    num_workers=num_workers,
)

# %% Define the model
model = torch.hub.load(
    "mateuszbuda/brain-segmentation-pytorch",
    "unet",
    in_channels=1,
    out_channels=len(classes),
    init_features=init_model_features,
    pretrained=False,
)

model = model.to(device)

# %% Define the optimizer
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

# %% Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Use custom loss function wrapper that handles NaN values in the target. This works with any PyTorch loss function
criterion = CellMapLossWrapper(criterion)

# %% Train the model
losses = np.empty((epochs * iterations_per_epoch))
validation_scores = np.empty(epochs)

# Training outer loop, across epochs
training_bar = trange(epochs, leave=True, position=0)
for epoch in training_bar:

    # Set the model to training mode to enable backpropagation
    model.train()

    # Refresh the train loader to shuffle the data yielded by the dataloader
    train_loader.refresh()

    # Training loop for the epoch
    epoch_bar = tqdm(train_loader.loader, leave=False, position=1)
    for i, (inputs, targets) in enumerate(epoch_bar):
        # Move the data to the device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the gradients, so that they don't accumulate across iterations
        optimizer.zero_grad()

        # Forward pass (compute the output of the model)
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Save the loss for logging
        losses[epoch * iterations_per_epoch + i] = loss.item()

        # Backward pass (compute the gradients)
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the progress bar
        epoch_bar.set_description(f"Loss: {loss.item():.4f}")

    # Save the model
    torch.save(
        model.state_dict(),
        model_save_path.format(epoch=epoch + 1, model_name=model_name),
    )

    # Set the model to evaluation mode to disable backpropagation
    model.eval()

    # Compute the validation score by averaging the loss across the validation set
    val_score = 0
    val_bar = tqdm(val_loader, leave=False, position=1)
    for inputs, targets in val_bar:
        outputs = model(inputs)
        val_score += criterion(outputs, targets).item()
    val_score /= len(val_loader)
    validation_scores[epoch] = val_score

    # Update the progress bar
    training_bar.set_description(f"Validation score: {val_score:.4f}")
    training_bar.refresh()

    # Generate and save some example figures from the validation set
    save_result_figs(
        inputs,
        outputs,
        targets,
        classes,
        figures_save_path.format(
            epoch=epoch + 1, model_name=model_name, label="{label}"
        ),
    )

# %% Plot the training loss and validation score
fig = get_loss_plot(losses, validation_scores, iterations_per_epoch)
fig.savefig(
    figures_save_path.format(epoch="summary", model_name=model_name, label="loss_plot")
)
