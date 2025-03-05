import os
import random
import time

import numpy as np
import torch
from cellmap_data.utils import get_fig_dict
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tensorboardX import SummaryWriter
from tqdm import tqdm
from upath import UPath
import matplotlib.pyplot as plt

from .models import ResNet, UNet_2D, UNet_3D, ViTVNet, load_best_val, load_latest
from .utils import (
    CellMapLossWrapper,
    get_dataloader,
    load_safe_config,
    make_datasplit_csv,
    make_s3_datasplit_csv,
)


def train(config_path: str):
    """
    Train a model using the configuration file at the specified path. The model checkpoints and training logs, as well as the datasets used for training, will be saved to the paths specified in the configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file to use for training the model. This file should be a Python file that defines the hyperparameters and other configurations for training the model. This may include:
        - model_save_path: Path to save the model checkpoints. Default is 'checkpoints/{model_name}_{epoch}.pth'.
        - logs_save_path: Path to save the logs for tensorboard. Default is 'tensorboard/{model_name}'. Training progress may be monitored by running `tensorboard --logdir <logs_save_path>` in the terminal.
        - datasplit_path: Path to the datasplit file that defines the train/val split the dataloader should use. Default is 'datasplit.csv'.
        - validation_prob: Proportion of the datasets to use for validation. This is used if the datasplit CSV specified by `datasplit_path` does not already exist. Default is 0.15.
        - learning_rate: Learning rate for the optimizer. Default is 0.0001.
        - batch_size: Batch size for the dataloader. Default is 8.
        - input_array_info: Dictionary containing the shape and scale of the input data. Default is {'shape': (1, 128, 128), 'scale': (8, 8, 8)}.
        - target_array_info: Dictionary containing the shape and scale of the target data. Default is to use `input_array_info`.
        - epochs: Number of epochs to train the model for. Default is 1000.
        - iterations_per_epoch: Number of iterations per epoch. Each iteration includes an independently generated random batch from the training set. Default is 1000.
        - random_seed: Random seed for reproducibility. Default is 42.
        - classes: List of classes to train the model to predict. This will be reflected in the data included in the datasplit, if generated de novo after calling this script. Default is ['nuc', 'er'].
        - model_name: Name of the model to use. If the config file constructs the PyTorch model, this name can be anything. If the config file does not construct the PyTorch model, the model_name will need to specify which included architecture to use. This includes '2d_unet', '2d_resnet', '3d_unet', '3d_resnet', and 'vitnet'. Default is '2d_unet'. See the `models` module `README.md` for more information.
        - model_to_load: Name of the pre-trained model to load. Default is the same as `model_name`.
        - model_kwargs: Dictionary of keyword arguments to pass to the model constructor. Default is {}. If the PyTorch `model` is passed, this will be ignored. See the `models` module `README.md` for more information.
        - model: PyTorch model to use for training. If this is provided, the `model_name` and `model_to_load` can be any string. Default is None.
        - load_model: Which model checkpoint to load if it exists. Options are 'latest' or 'best'. If no checkpoints exist, will silently use the already initialized model. Default is 'latest'.
        - spatial_transforms: Dictionary of spatial transformations to apply to the training data. Default is {'mirror': {'axes': {'x': 0.5, 'y': 0.5}}, 'transpose': {'axes': ['x', 'y']}, 'rotate': {'axes': {'x': [-180, 180], 'y': [-180, 180]}}}. See the `dataloader` module documentation for more information.
        - validation_time_limit: Maximum time to spend on validation in seconds. If None, there is no time limit. Default is None.
        - validation_batch_limit: Maximum number of validation batches to process. If None, there is no limit. Default is None.
        - device: Device to use for training. If None, will use 'cuda' if available, 'mps' if available, or 'cpu' otherwise. Default is None.
        - use_s3: Whether to use the S3 bucket for the datasplit. Default is False.
        - optimizer: PyTorch optimizer to use for training. Default is `torch.optim.RAdam(model.parameters(), lr=learning_rate, decoupled_weight_decay=True)`.
        - criterion: Uninstantiated PyTorch loss function to use for training. Default is `torch.nn.BCEWithLogitsLoss`.
        - criterion_kwargs: Dictionary of keyword arguments to pass to the loss function constructor. Default is {}.
        - weight_loss: Whether to weight the loss function by class counts found in the datasets. Default is True.
        - use_mutual_exclusion: Whether to use mutual exclusion to infer labels for unannotated pixels. Default is False.
        - weighted_sampler: Whether to use a sampler weighted by class counts for the dataloader. Default is True.
        - train_raw_value_transforms: Transform to apply to the raw values for training. Defaults to T.Compose([T.ToDtype(torch.float, scale=True), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]) which normalizes the input data, converts it to float32, and replaces NaNs with 0. This can be used to add augmentations such as random erasing, blur, noise, etc.
        - val_raw_value_transforms: Transform to apply to the raw values for validation, similar to `train_raw_value_transforms`. Default is the same as `train_raw_value_transforms`.
        - target_value_transforms: Transform to apply to the target values. Default is T.Compose([T.ToDtype(torch.float), Binarize()]) which converts the input masks to float32 and threshold at 0 (turning object ID's into binary masks for use with binary cross entropy loss). This can be used to specify other targets, such as distance transforms.
        - max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is performed. Default is None. This can be useful to prevent exploding gradients which would lead to NaNs in the weights.
        - force_all_classes: Whether to force all classes to be present in each batch provided by dataloaders. Can either be `True` to force this for both validation and training dataloader, `False` to force for neither, or `train` / `validate` to restrict it to training or validation, respectively. Default is 'validate'.
        - scheduler: PyTorch learning rate scheduler (or uninstantiated class) to use for training. Default is None. If provided, the scheduler will be called at the end of each epoch.
        - scheduler_kwargs: Dictionary of keyword arguments to pass to the scheduler constructor. Default is {}. If `scheduler` instantiation is provided, this will be ignored.

    Returns
    -------
    None

    """
    # Pick the fastest algorithm for the hardware
    torch.backends.cudnn.benchmark = True

    # %% Load the configuration file
    config = UPath(config_path).stem
    config = load_safe_config(config_path)
    # %% Set hyperparameters and other configurations from the config file
    model_save_path = getattr(
        config, "model_save_path", UPath("checkpoints/{model_name}_{epoch}.pth").path
    )
    logs_save_path = getattr(
        config, "logs_save_path", UPath("tensorboard/{model_name}").path
    )
    datasplit_path = getattr(config, "datasplit_path", "datasplit.csv")
    validation_prob = getattr(config, "validation_prob", 0.15)
    learning_rate = getattr(config, "learning_rate", 0.0001)
    batch_size = getattr(config, "batch_size", 8)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    epochs = getattr(config, "epochs", 1000)
    iterations_per_epoch = getattr(config, "iterations_per_epoch", 1000)
    random_seed = getattr(config, "random_seed", getattr(os.environ, "SEED", 42))
    classes = getattr(config, "classes", ["nuc", "er"])
    model_name = getattr(config, "model_name", "2d_unet")
    model_to_load = getattr(config, "model_to_load", model_name)
    model_kwargs = getattr(config, "model_kwargs", {})
    model = getattr(config, "model", None)
    load_model = getattr(config, "load_model", "latest")
    spatial_transforms = getattr(
        config,
        "spatial_transforms",
        {
            "mirror": {"axes": {"x": 0.5, "y": 0.5}},
            "transpose": {"axes": ["x", "y"]},
            "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
        },
    )
    validation_time_limit = getattr(config, "validation_time_limit", None)
    validation_batch_limit = getattr(config, "validation_batch_limit", None)
    device = getattr(config, "device", None)
    use_s3 = getattr(config, "use_s3", False)
    use_mutual_exclusion = getattr(config, "use_mutual_exclusion", False)
    weighted_sampler = getattr(config, "weighted_sampler", True)
    train_raw_value_transforms = getattr(
        config,
        "train_raw_value_transforms",
        T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
    )
    val_raw_value_transforms = getattr(
        config,
        "val_raw_value_transforms",
        T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
    )
    target_value_transforms = getattr(
        config,
        "target_value_transforms",
        T.Compose([T.ToDtype(torch.float), Binarize()]),
    )
    max_grad_norm = getattr(config, "max_grad_norm", None)
    force_all_classes = getattr(config, "force_all_classes", "validate")

    # %% Define the optimizer, from the config file or default to RAdam
    optimizer = getattr(
        config,
        "optimizer",
        torch.optim.RAdam(
            model.parameters(), lr=learning_rate, decoupled_weight_decay=True
        ),
    )

    # %% Define the scheduler, from the config file or default to None
    scheduler = getattr(config, "scheduler", None)
    if isinstance(scheduler, type):
        scheduler_kwargs = getattr(config, "scheduler_kwargs", {})
        scheduler = scheduler(optimizer, **scheduler_kwargs)

    # %% Define the loss function, from the config file or default to BCEWithLogitsLoss
    criterion = getattr(config, "criterion", torch.nn.BCEWithLogitsLoss)
    criterion_kwargs = getattr(config, "criterion_kwargs", {})
    weight_loss = getattr(config, "weight_loss", True)

    # %% Make sure the save path exists
    for path in [model_save_path, logs_save_path, datasplit_path]:
        dirpath = os.path.dirname(path)
        if len(dirpath) > 0:
            os.makedirs(dirpath, exist_ok=True)

    # %% Set the random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # %% Check that the GPU is available
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Training device: {device}")

    # %% Make the datasplit file if it doesn't exist
    if not os.path.exists(datasplit_path):
        if use_s3:
            make_s3_datasplit_csv(
                classes=classes,
                csv_path=datasplit_path,
                validation_prob=validation_prob,
                force_all_classes=force_all_classes,
            )
        else:
            make_datasplit_csv(
                classes=classes,
                csv_path=datasplit_path,
                validation_prob=validation_prob,
                force_all_classes=force_all_classes,
            )

    # %% Download the data and make the dataloader
    train_loader, val_loader = get_dataloader(
        datasplit_path,
        classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=iterations_per_epoch,
        random_validation=validation_time_limit or validation_batch_limit,
        device=device,
        weighted_sampler=weighted_sampler,
        use_mutual_exclusion=use_mutual_exclusion,
        train_raw_value_transforms=train_raw_value_transforms,
        val_raw_value_transforms=val_raw_value_transforms,
        target_value_transforms=target_value_transforms,
    )

    # %% If no model is provided, create a new model
    if model is None:
        if "2d" in model_name.lower():
            if "unet" in model_name.lower():
                model = UNet_2D(1, len(classes), **model_kwargs)
            elif "resnet" in model_name.lower():
                model = ResNet(ndims=2, output_nc=len(classes), **model_kwargs)
            else:
                raise ValueError(
                    f"Unknown model name: {model_name}. Preconfigured 2D models are '2d_unet' and '2d_resnet'."
                )
        elif "3d" in model_name.lower():
            if "unet" in model_name.lower():
                model = UNet_3D(1, len(classes), **model_kwargs)
            elif "resnet" in model_name.lower():
                model = ResNet(ndims=3, output_nc=len(classes), **model_kwargs)
            else:
                raise ValueError(
                    f"Unknown model name: {model_name}. Preconfigured 3D models are '3d_unet' and '3d_resnet', or 'vitnet'."
                )
        elif "vitnet" in model_name.lower():
            model = ViTVNet(len(classes), **model_kwargs)
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Preconfigured models are '2d_unet', '2d_resnet', '3d_unet', '3d_resnet', and 'vitnet'. Otherwise provide a custom model as a torch.nn.Module."
            )

    # Optionally, load a pre-trained model
    if load_model.lower() == "latest":
        # Check to see if there are any checkpoints and if so load the latest one
        # Use the command below for loading the latest model, otherwise comment it out
        load_latest(model_save_path.format(epoch="*", model_name=model_to_load), model)
    elif load_model.lower() == "best":
        # Load the checkpoint with the best validation score
        # Use the command below for loading the epoch with the best validation score, otherwise comment it out
        load_best_val(
            logs_save_path.format(model_name=model_to_load),
            model_save_path.format(epoch="{epoch}", model_name=model_to_load),
            model,
        )

    # %% Move model to device
    model = model.to(device)

    # Use custom loss function wrapper that handles NaN values in the target. This works with any PyTorch loss function
    if weight_loss:
        pos_weight = list(train_loader.dataset.class_weights.values())
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device).flatten()

        # Adjust weight for data dimensions
        pos_weight = pos_weight[:, None, None]
        if all([s > 1 for s in target_array_info["shape"]]):
            pos_weight = pos_weight[..., None]
        criterion_kwargs["pos_weight"] = pos_weight
    criterion = CellMapLossWrapper(criterion, **criterion_kwargs)

    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())

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

        # Refresh the train loader to shuffle the data yielded by the dataloader
        train_loader.refresh()

        # epoch_bar = tqdm(train_loader.loader, desc="Training", dynamic_ncols=True)
        # for batch in epoch_bar:
        loader = iter(train_loader.loader)
        epoch_bar = tqdm(
            range(iterations_per_epoch), desc="Training", dynamic_ncols=True
        )
        for _ in epoch_bar:
            # for some reason this seems to be faster...
            batch = next(loader)

            # Increment the training iteration
            n_iter += 1

            # Zero the gradients, so that they don't accumulate across iterations
            optimizer.zero_grad()

            # Forward pass (compute the output of the model)
            if len(input_keys) > 1:
                inputs = {key: batch[key] for key in input_keys}
            else:
                inputs = batch[input_keys[0]]
                # Assumes the model input is a single tensor
            outputs = model(inputs)

            # Compute the loss
            if len(target_keys) > 1:
                targets = {key: batch[key] for key in target_keys}
            else:
                targets = batch[target_keys[0]]
                # Assumes the model output is a single tensor
            loss = criterion(outputs, targets)

            # Backward pass (compute the gradients)
            loss.backward()

            # Clip the gradients if necessary
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update the weights
            optimizer.step()

            # Update the progress bar
            post_fix_dict["Loss"] = f"{loss.item()}"
            epoch_bar.set_postfix(post_fix_dict)

            # Log the loss using tensorboard
            writer.add_scalar("loss", loss.item(), n_iter)

        if scheduler is not None:
            # Step the scheduler at the end of each epoch
            scheduler.step()
            writer.add_scalar("lr", scheduler.get_last_lr()[0], n_iter)

        # Save the model
        torch.save(
            model.state_dict(),
            model_save_path.format(epoch=epoch + 1, model_name=model_name),
        )

        # Compute the validation score by averaging the loss across the validation set
        if len(val_loader.loader) > 0:
            val_score = 0
            val_loader.refresh()
            if validation_time_limit is not None:
                elapsed_time = 0
                last_time = time.time()
                val_bar = val_loader.loader
                pbar = tqdm(
                    total=validation_time_limit,
                    desc="Validation",
                    bar_format="{l_bar}{bar}| {remaining}s remaining",
                    dynamic_ncols=True,
                )
            else:
                val_bar = tqdm(
                    val_loader.loader,
                    desc="Validation",
                    total=validation_batch_limit or len(val_loader.loader),
                    dynamic_ncols=True,
                )

            # Free up GPU memory, disable backprop etc.
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            model.eval()

            # Validation loop
            with torch.no_grad():
                i = 0
                for batch in val_bar:
                    if len(input_keys) > 1:
                        inputs = {key: batch[key] for key in input_keys}
                    else:
                        inputs = batch[input_keys[0]]
                    outputs = model(inputs)

                    # Compute the loss
                    if len(target_keys) > 1:
                        targets = {key: batch[key] for key in target_keys}
                    else:
                        targets = batch[target_keys[0]]
                    val_score += criterion(outputs, targets).item()
                    i += 1

                    # Check time limit
                    if validation_time_limit is not None:
                        last_elapsed_time = time.time() - last_time
                        elapsed_time += last_elapsed_time
                        if elapsed_time >= validation_time_limit:
                            break
                        pbar.update(last_elapsed_time)
                        last_time = time.time()
                    # Check batch limit
                    elif (
                        validation_batch_limit is not None
                        and i >= validation_batch_limit
                    ):
                        break
                val_score /= i

                # Log the validation using tensorboard
                writer.add_scalar("validation", val_score, n_iter)

                # Update the progress bar
                post_fix_dict["Validation"] = f"{val_score:.4f}"

        # Generate and save figures from the last batch of the validation to appear in tensorboard
        # TODO: Make this more general rather than only taking the first key
        if isinstance(outputs, dict):
            outputs = list(outputs.values())[0]
        if isinstance(inputs, dict):
            inputs = list(inputs.values())[0]
        if isinstance(targets, dict):
            targets = list(targets.values())[0]
        figs = get_fig_dict(inputs, targets, outputs, classes)
        for name, fig in figs.items():
            writer.add_figure(name, fig, n_iter)
            plt.close(fig)

        # Clear the GPU memory again
        torch.cuda.empty_cache()

    # Close the summarywriter
    writer.close()

    # %%
