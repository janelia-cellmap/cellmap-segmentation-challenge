import torch
from torch.utils.data import DataLoader
from typing import Mapping, Optional, Sequence

from cellmap_data import CellMapDataSplit, CellMapDataLoader
from cellmap_data.transforms.augment import (
    Normalize,
    NaNtoNum,
)
import torchvision.transforms.v2 as T


def get_dataloader(
    datasplit_path: str,
    classes: Sequence[str],
    batch_size: int,
    array_info: Optional[Mapping[str, Sequence[int | float]]] = None,
    input_array_info: Optional[Mapping[str, Sequence[int | float]]] = None,
    target_array_info: Optional[Mapping[str, Sequence[int | float]]] = None,
    iterations_per_epoch: int = 1000,
    device: str | torch.device = "cuda",
) -> tuple[CellMapDataLoader, DataLoader]:
    """
    Get the train and validation dataloaders.

    This function gets the train and validation dataloaders for the given datasplit file, classes, batch size, array
    info, iterations per epoch, number of workers, and device.

    Parameters
    ----------
    datasplit_path : str
        Path to the datasplit file that defines the train/val split the dataloader should use.
    classes : Sequence[str]
        List of classes to segment.
    batch_size : int
        Batch size for the dataloader.
    array_info : Optional[Mapping[str, Sequence[int | float]]]
        Dictionary containing the shape and scale of the data to load for the input and target. Either `array_info` or `input_array_info` & `target_array_info` must be provided.
    input_array_info : Optional[Mapping[str, Sequence[int | float]]]
        Dictionary containing the shape and scale of the data to load for the input.
    target_array_info : Optional[Mapping[str, Sequence[int | float]]]
        Dictionary containing the shape and scale of the data to load for the target.
    iterations_per_epoch : int
        Number of iterations per epoch.
    device : str or torch.device
        Device to use for the dataloaders.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Tuple containing the train and validation dataloaders.
    """

    input_arrays = {
        "input": input_array_info if input_array_info is not None else array_info
    }
    target_arrays = {
        "output": target_array_info if target_array_info is not None else array_info
    }

    value_transforms = T.Compose(
        [
            Normalize(),
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0.0, "posinf": None, "neginf": None}),
        ],
    )

    datasplit = CellMapDataSplit(
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        classes=classes,
        pad=True,
        csv_path=datasplit_path,
        train_raw_value_transforms=value_transforms,
        val_raw_value_transforms=value_transforms,
    )

    validation_loader = CellMapDataLoader(
        datasplit.validation_blocks.to(device),
        classes=classes,
        batch_size=batch_size,
        is_train=False,
    ).loader

    train_loader = CellMapDataLoader(
        datasplit.train_datasets_combined.to(device),
        classes=classes,
        batch_size=batch_size,
        sampler=lambda: datasplit.train_datasets_combined.get_subset_random_sampler(
            iterations_per_epoch * batch_size, weighted=False
        ),
    )

    return train_loader, validation_loader  # type: ignore
