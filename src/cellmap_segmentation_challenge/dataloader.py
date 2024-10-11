import torch
from torch.utils.data import DataLoader
from typing import Mapping, Optional, Sequence, Any
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
    spatial_transforms: Optional[Mapping[str, Any]] = None,
    # TODO: Add value transforms
    iterations_per_epoch: int = 1000,
    device: str | torch.device = "cuda",
) -> tuple[CellMapDataLoader, DataLoader]:
    """
    Get the train and validation dataloaders.

    This function gets the train and validation dataloaders for the given datasplit file, classes, batch size, array
    info, spatial transforms, iterations per epoch, number of workers, and device.

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
    spatial_transforms : Optional[Mapping[str, any]]
        Dictionary containing the spatial transformations to apply to the data.
        For example the dictionary could contain transformations like mirror, transpose, and rotate.

    spatial_transforms = {
          # 3D

           # Probability of applying mirror for each axis
           # Values range from 0 (no mirroring) to 1 (will always mirror)
          "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.5}},

           # Specifies the axes that will be invovled in the trasposition
          "transpose": {"axes": ["x", "y", "z"]},

           # Defines rotation range for each axis.
           # Rotation angle for each axis is randomly chosen within the specified range (-180, 180).
          "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},

          # 2D (used when there is no z axis)
          # "mirror": {"axes": {"x": 0.5, "y": 0.5}},
          # "transpose": {"axes": ["x", "y"]},
          # "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
    }


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
    assert (
        input_arrays is not None and target_arrays is not None
    ), "No array info provided"

    value_transforms = T.Compose(
        [
            Normalize(),
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ],
    )

    datasplit = CellMapDataSplit(
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        classes=classes,
        pad=False,  # TODO: Make work with padding
        csv_path=datasplit_path,
        train_raw_value_transforms=value_transforms,
        val_raw_value_transforms=value_transforms,
        target_value_transforms=T.ToDtype(torch.float),
        spatial_transforms=spatial_transforms,
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
