from torch.utils.data import DataLoader
from typing import Mapping, Sequence

from cellmap_data import CellMapDataSplit, CellMapDataLoader


def get_dataloader(
    datasplit_path: str,
    classes: Sequence[str],
    batch_size: int,
    array_info: Mapping[str, Sequence[int | float]],
    iterations_per_epoch: int,
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
    array_info : Mapping[Sequence[int], Sequence[float | int]]
        Shape and voxel size of the data to load.
    iterations_per_epoch : int
        Number of iterations per epoch.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Tuple containing the train and validation dataloaders.
    """

    input_arrays = {"input": array_info}
    target_arrays = {"output": array_info}

    datasplit = CellMapDataSplit(
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        classes=classes,
        pad=True,
        csv_path=datasplit_path,
    )

    validation_loader = CellMapDataLoader(
        datasplit.validation_blocks,
        classes=classes,
        batch_size=batch_size,
        is_train=False,
    ).loader

    train_loader = CellMapDataLoader(
        datasplit.train_datasets_combined,
        classes=classes,
        batch_size=batch_size,
        sampler=lambda: datasplit.train_datasets_combined.get_subset_random_sampler(
            iterations_per_epoch * batch_size, weighted=False
        ),
    )

    return train_loader, validation_loader  # type: ignore
