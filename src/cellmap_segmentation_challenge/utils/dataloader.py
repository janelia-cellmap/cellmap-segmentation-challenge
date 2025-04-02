from typing import Any, Mapping, Optional, Sequence
import functools

import torch
import torchvision.transforms.v2 as T
from cellmap_data import CellMapDataLoader, CellMapDataSplit
from cellmap_data.transforms.augment import NaNtoNum, Normalize, Binarize
from cellmap_segmentation_challenge.utils import get_class_relations


def get_dataloader(
    datasplit_path: str,
    classes: Sequence[str] | None,
    batch_size: int,
    input_array_info: Optional[Mapping[str, Sequence[int | float]]] = None,
    target_array_info: Optional[Mapping[str, Sequence[int | float]]] = None,
    spatial_transforms: Optional[Mapping[str, Any]] = None,
    target_value_transforms: Optional[T.Transform] = T.Compose(
        [T.ToDtype(torch.float), Binarize()]
    ),
    train_raw_value_transforms: Optional[T.Transform] = T.Compose(
        [
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ],
    ),
    val_raw_value_transforms: Optional[T.Transform] = T.Compose(
        [
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ],
    ),
    iterations_per_epoch: int = 1000,
    random_validation: bool = False,
    device: Optional[str | torch.device] = None,
    use_mutual_exclusion: bool = False,
    weighted_sampler: bool = True,
    **kwargs,
) -> tuple[CellMapDataLoader, CellMapDataLoader | None]:
    """
    Get the train and validation dataloaders.

    This function gets the train and validation dataloaders for the given datasplit file, classes, batch size, array
    info, spatial transforms, iterations per epoch, number of workers, and device.

    Parameters
    ----------
    datasplit_path : str
        Path to the datasplit file that defines the train/val split the dataloader should use.
    classes : Sequence[str] | None
        List of classes to segment. If None, assumes training on raw data.
    batch_size : int
        Batch size for the dataloader.
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
    target_value_transforms : Optional[torchvision.transforms.v2.Transform]
        Transform to apply to the target values. Defaults to T.Compose([T.ToDtype(torch.float), Binarize()]) which converts the input masks to float32 and threshold at 0 (turning object ID's into binary masks for use with binary cross entropy loss).
    train_raw_value_transforms : Optional[torchvision.transforms.v2.Transform]
        Transform to apply to the raw values for training. Defaults to T.Compose([Normalize(), T.ToDtype(torch.float, scale=True), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]) which normalizes the input data, converts it to float32, and replaces NaNs with 0. This can be used to add augmentations such as random erasing, blur, noise, etc.
    val_raw_value_transforms : Optional[torchvision.transforms.v2.Transform]
        Transform to apply to the raw values for validation. Defaults to T.Compose([Normalize(), T.ToDtype(torch.float, scale=True), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]) which normalizes the input data, converts it to float32, and replaces NaNs with 0.
    iterations_per_epoch : int
        Number of iterations per epoch.
    random_validation : bool
        Whether or not to randomize the validation data draws. Useful if not evaluating on the entire validation set everytime. Defaults to False.
    device : Optional[str or torch.device]
        Device to use for training. If None, defaults to "cuda" if available, or "mps" if available, or "cpu".
    use_mutual_exclusion : bool
        Whether to use mutually exclusive class labels to infer non-present labels for the training data. Defaults to False.
    weighted_sampler : bool
        Whether to weight sample draws based on the number of positive labels within a dataset. Defaults to True.
    **kwargs : Any
        Additional keyword arguments to pass to the CellMapDataLoader.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Tuple containing the train and validation dataloaders.
    """
    # If "shape" and "scale" are not top level keys in the array_info dict, then we can assume a full dictionary of input arrays are being passed in
    if (
        "shape" in input_array_info
        and "scale" in input_array_info
        and len(input_array_info.keys()) == 2
    ):
        input_arrays = {"input": input_array_info}
    else:
        input_arrays = input_array_info

    if target_array_info is not None and (
        "shape" in target_array_info
        and "scale" in target_array_info
        and len(target_array_info.keys()) == 2
    ):
        target_arrays = {"output": target_array_info}
    else:
        target_arrays = target_array_info

    assert input_arrays is not None, "No array info provided"

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if use_mutual_exclusion:
        class_relation_dict = get_class_relations(named_classes=classes)
    else:
        class_relation_dict = None

    datasplit = CellMapDataSplit(
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        classes=classes,
        pad=True,
        csv_path=datasplit_path,
        train_raw_value_transforms=train_raw_value_transforms,
        val_raw_value_transforms=val_raw_value_transforms,
        target_value_transforms=target_value_transforms,
        spatial_transforms=spatial_transforms,
        device=device,
        class_relation_dict=class_relation_dict,
    )

    if len(datasplit.validation_datasets) >= 0:
        _kwargs = {
            "classes": classes,
            "batch_size": batch_size,
            "is_train": random_validation,
            "device": device,
        }
        _kwargs.update(kwargs)
        validation_loader = CellMapDataLoader(
            datasplit.validation_blocks.to(device),
            **_kwargs,
        )
    else:
        validation_loader = None

    _kwargs = {
        "classes": classes,
        "batch_size": batch_size,
        "sampler": functools.partial(
            datasplit.train_datasets_combined.get_subset_random_sampler,
            num_samples=iterations_per_epoch * batch_size,
            weighted=weighted_sampler,
        ),
        "device": device,
        "is_train": True,
    }
    _kwargs.update(kwargs)
    train_loader = CellMapDataLoader(
        datasplit.train_datasets_combined.to(device),
        **_kwargs,
    )

    return train_loader, validation_loader  # type: ignore
