import torch
from typing import Mapping, Sequence

import cellmap_data


def get_dataloader(
    datasplit_path: str,
    classes: Sequence[str],
    batch_size: int,
    array_info: Mapping[str, Sequence[int | float]],
    iterations_per_epoch: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
    array_info : Mapping[Sequence[int], Sequence[float | int]]
        Shape and voxel size of the data to load.
    iterations_per_epoch : int
        Number of iterations per epoch.
    num_workers : int
        Number of workers for the dataloader to use.
    device : torch.device
        Device to use for the dataloader.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Tuple containing the train and validation dataloaders.
    """
    #
