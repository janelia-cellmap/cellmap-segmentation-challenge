import copy
import os
import tempfile
from glob import glob
from typing import Any

import torch
import torchvision.transforms.v2 as T
from cellmap_data import CellMapDatasetWriter, CellMapImage
from cellmap_data.utils import (
    array_has_singleton_dim,
    is_array_2D,
    permute_singleton_dimension,
)
from cellmap_data.transforms.augment import NaNtoNum
from tqdm import tqdm
from upath import UPath

from .config import CROP_NAME, PREDICTIONS_PATH, RAW_NAME, SEARCH_PATH
from .models import get_model
from .utils import (
    load_safe_config,
    get_test_crops,
    get_test_crop_labels,
    get_data_from_batch,
    get_singleton_dim,
    squeeze_singleton_dim,
    structure_model_output,
    unsqueeze_singleton_dim,
)
from .utils.datasplit import get_formatted_fields, get_raw_path


def predict_orthoplanes(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    print("Predicting orthogonal planes.")

    # Make a temporary prediction for each axis
    tmp_dir = tempfile.TemporaryDirectory()
    print(f"Temporary directory for predictions: {tmp_dir.name}")
    for axis in range(3):
        # Actually slice per axis by permuting singleton dimension
        temp_kwargs = dataset_writer_kwargs.copy()
        temp_kwargs["target_path"] = os.path.join(
            tmp_dir.name, "output.zarr", str(axis)
        )
        # Permute input_arrays and target_arrays so singleton is at the current axis
        input_arrays = {k: v.copy() for k, v in temp_kwargs["input_arrays"].items()}
        target_arrays = {k: v.copy() for k, v in temp_kwargs["target_arrays"].items()}
        permute_singleton_dimension(input_arrays, axis)
        permute_singleton_dimension(target_arrays, axis)
        temp_kwargs["input_arrays"] = input_arrays
        temp_kwargs["target_arrays"] = target_arrays
        _predict(
            model,
            temp_kwargs,
            batch_size=batch_size,
        )

    # Get dataset writer for the average of predictions from x, y, and z orthogonal planes
    # TODO: Skip loading raw data
    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)

    # Load the images for the individual predictions
    single_axis_images = {
        array_name: {
            label: [
                CellMapImage(
                    os.path.join(tmp_dir.name, "output.zarr", str(axis), label),
                    target_class=label,
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for axis in range(3)
            ]
            for label in dataset_writer_kwargs["classes"]
        }
        for array_name, array_info in dataset_writer_kwargs["target_arrays"].items()
    }

    # Combine the predictions from the x, y, and z orthogonal planes
    print("Combining predictions.")
    for batch in tqdm(dataset_writer.loader(batch_size=batch_size), dynamic_ncols=True):
        # For each class, get the predictions from the x, y, and z orthogonal planes
        outputs = {}
        for array_name, images in single_axis_images.items():
            outputs[array_name] = {}
            for label in dataset_writer_kwargs["classes"]:
                outputs[array_name][label] = []
                for idx in batch["idx"]:
                    average_prediction = []
                    for image in images[label]:
                        average_prediction.append(image[dataset_writer.get_center(idx)])
                    average_prediction = torch.stack(average_prediction).mean(dim=0)
                    outputs[array_name][label].append(average_prediction)
                outputs[array_name][label] = torch.stack(outputs[array_name][label])

        # Save the outputs
        dataset_writer[batch["idx"]] = outputs

    tmp_dir.cleanup()


def _predict(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    """
    Predicts the output of a model on a large dataset by splitting it into blocks and predicting each block separately.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction.
    dataset_writer_kwargs : dict[str, Any]
        A dictionary containing the arguments for the dataset writer.
    batch_size : int
        The batch size to use for prediction
    """

    model.eval()
    device = dataset_writer_kwargs["device"]
    input_keys = list(dataset_writer_kwargs["input_arrays"].keys())
    
    # Get the classes to use for model output (all classes the model was trained on)
    # vs the classes to actually save (filtered by test_crop_manifest)
    model_classes = dataset_writer_kwargs.get("model_classes", dataset_writer_kwargs["classes"])
    classes_to_save = dataset_writer_kwargs["classes"]

    # Test a single batch to get number of output channels
    test_batch = {
        k: torch.rand((1, *info["shape"])).unsqueeze(0).to(device)
        for k, info in dataset_writer_kwargs["input_arrays"].items()
    }
    test_inputs = get_data_from_batch(test_batch, input_keys, device)
    # Apply the same singleton-dimension squeezing as in the main prediction loop
    singleton_dim = get_singleton_dim(
        list(dataset_writer_kwargs["input_arrays"].values())[0]["shape"]
    )
    if singleton_dim is not None:
        test_inputs = squeeze_singleton_dim(test_inputs, singleton_dim + 1)
    with torch.no_grad():
        test_outputs = model(test_inputs)
    model_returns_class_dict = False
    num_channels_per_class = None
    if isinstance(test_outputs, dict):
        if set(test_outputs.keys()) == set(model_classes):
            # Keys are the class names; values are already per-class tensors
            model_returns_class_dict = True
        else:
            # Dict with non-class keys (e.g., resolution levels): use the first
            # value tensor to detect the channel count
            test_outputs = next(iter(test_outputs.values()))
    if not model_returns_class_dict and test_outputs.shape[1] > len(model_classes):
        if test_outputs.shape[1] % len(model_classes) == 0:
            num_channels_per_class = test_outputs.shape[1] // len(model_classes)
            # To avoid mutating the input dictionary (which may be shared across multiple
            # prediction calls), create a deep copy of target_arrays and update the shape
            # to include the channel dimension.
            target_arrays_copy = copy.deepcopy(dataset_writer_kwargs["target_arrays"])
            for key in target_arrays_copy.keys():
                current_shape = target_arrays_copy[key]["shape"]
                # Use the first input array's shape to determine expected spatial rank
                # (all input arrays should have the same spatial dimensions)
                first_input_key = next(iter(dataset_writer_kwargs["input_arrays"]))
                expected_spatial_rank = len(
                    dataset_writer_kwargs["input_arrays"][first_input_key]["shape"]
                )
                # Only prepend the channel dimension if the shape doesn't already include it
                # We check if the current rank matches the expected spatial rank (no channel dim yet)
                if len(current_shape) == expected_spatial_rank:
                    target_arrays_copy[key]["shape"] = (
                        num_channels_per_class,
                        *current_shape,
                    )
            # Replace target_arrays in the kwargs with the modified copy
            dataset_writer_kwargs = {
                **dataset_writer_kwargs,
                "target_arrays": target_arrays_copy,
            }
        else:
            raise ValueError(
                f"Number of output channels ({test_outputs.shape[1]}) does not match number of "
                f"classes ({len(model_classes)}). Should be a multiple of the "
                "number of classes."
            )
    del test_batch, test_inputs, test_outputs

    if "raw_value_transforms" not in dataset_writer_kwargs:
        dataset_writer_kwargs["raw_value_transforms"] = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        )

    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)
    dataloader = dataset_writer.loader(batch_size=batch_size)

    # Find singleton dimension if there is one
    # Only the first singleton dimension will be used for squeezing/unsqueezing.
    # If there are multiple singleton dimensions, only the first is handled.
    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):
            # Get the inputs, handling dict vs. tensor data
            inputs = get_data_from_batch(batch, input_keys, device)
            if singleton_dim is not None:
                inputs = squeeze_singleton_dim(inputs, singleton_dim + 2)
            outputs = model(inputs)
            if singleton_dim is not None:
                outputs = unsqueeze_singleton_dim(outputs, singleton_dim + 2)

            outputs = structure_model_output(
                outputs,
                model_classes,
                num_channels_per_class,
            )
            
            # Filter outputs to only include the classes that should be saved
            if model_classes != classes_to_save:
                filtered_outputs = {}
                for array_name, class_outputs in outputs.items():
                    if isinstance(class_outputs, dict):
                        # Filter to only include classes_to_save
                        filtered_outputs[array_name] = {
                            class_name: class_tensor
                            for class_name, class_tensor in class_outputs.items()
                            if class_name in classes_to_save
                        }
                    else:
                        # If it's not a dict (just a tensor), we need to index the tensor
                        # This assumes the tensor has shape (B, C, ...) where C corresponds to model_classes
                        # We need to select only the channels for classes_to_save
                        # classes_to_save should be a subset of model_classes by design
                        class_indices = [model_classes.index(c) for c in classes_to_save]
                        filtered_outputs[array_name] = class_outputs[:, class_indices, ...]
                outputs = filtered_outputs

            # Save the outputs
            dataset_writer[batch["idx"]] = outputs


def predict(
    config_path: str,
    crops: str = "test",
    output_path: str = PREDICTIONS_PATH,
    do_orthoplanes: bool = True,
    overwrite: bool = False,
    search_path: str = SEARCH_PATH,
    raw_name: str = RAW_NAME,
    crop_name: str = CROP_NAME,
):
    """
    Given a model configuration file and list of crop numbers, predicts the output of a model on a large dataset by splitting it into blocks and predicting each block separately.

    Parameters
    ----------
    config_path : str
        The path to the model configuration file. This can be the same as the config file used for training.
    crops: str, optional
        A comma-separated list of crop numbers to predict on, or "test" to predict on the entire test set. Default is "test".
        When crops="test", only the labels specified in the test_crop_manifest for each crop will be saved.
    output_path: str, optional
        The path to save the output predictions to, formatted as a string with a placeholders for the dataset, crop number, and label. Default is PREDICTIONS_PATH set in `cellmap-segmentation/config.py`.
    do_orthoplanes: bool, optional
        Whether to compute the average of predictions from x, y, and z orthogonal planes for the full 3D volume. This is sometimes called 2.5D predictions. It expects a model that yields 2D outputs. Similarly, it expects the input shape to the model to be 2D. Default is True for 2D models.
    overwrite: bool, optional
        Whether to overwrite the output dataset if it already exists. Default is False.
    search_path: str, optional
        The path to search for the raw dataset, with placeholders for dataset and name. Default is SEARCH_PATH set in `cellmap-segmentation/config.py`.
    raw_name: str, optional
        The name of the raw dataset. Default is RAW_NAME set in `cellmap-segmentation/config.py`.
    crop_name: str, optional
        The name of the crop dataset with placeholders for crop and label. Default is CROP_NAME set in `cellmap-segmentation/config.py`.
        
    Notes
    -----
    When crops="test", the function will only save predictions for labels that are specified 
    in the test_crop_manifest for each specific crop. This ensures that only the labels that 
    will be scored are saved, reducing storage requirements and processing time.
    """
    config = load_safe_config(config_path)
    classes = config.classes
    batch_size = getattr(config, "batch_size", 8)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    value_transforms = getattr(
        config,
        "value_transforms",
        T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
    )
    model = config.model

    # %% Check that the GPU is available
    if getattr(config, "device", None) is not None:
        device = config.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Prediction device: {device}")

    # %% Move model to device
    model = model.to(device)

    # Optionally, load a pre-trained model
    checkpoint_epoch = get_model(config)
    if checkpoint_epoch is not None:
        print(f"Loaded model checkpoint from epoch: {checkpoint_epoch}")

    if do_orthoplanes and (
        array_has_singleton_dim(input_array_info)
        or is_array_2D(input_array_info, summary=any)
    ):
        # If the model is a 2D model, compute the average of predictions from x, y, and z orthogonal planes
        predict_func = predict_orthoplanes
    elif is_array_2D(input_array_info, summary=any) or is_array_2D(
        target_array_info, summary=any
    ):
        if is_array_2D(input_array_info, summary=any):
            permute_singleton_dimension(input_array_info, axis=0)
        if is_array_2D(target_array_info, summary=any):
            permute_singleton_dimension(target_array_info, axis=0)
        print(
            "Warning: Model appears to be 2D, but do_orthoplanes is set to False. Predictions will be made only on z slices."
        )
        predict_func = _predict
    else:
        predict_func = _predict

    assert (
        input_array_info is not None and target_array_info is not None
    ), "No array info provided"
    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}

    # Get the crops to predict on
    if crops == "test":
        test_crops = get_test_crops()
        dataset_writers = []
        for crop in test_crops:
            # Get path to raw dataset
            raw_path = search_path.format(dataset=crop.dataset, name=raw_name)

            # Get the boundaries of the crop
            target_bounds = {
                "output": {
                    axis: [
                        crop.gt_source.translation[i],
                        crop.gt_source.translation[i]
                        + crop.gt_source.voxel_size[i] * crop.gt_source.shape[i],
                    ]
                    for i, axis in enumerate("zyx")
                },
            }

            # Get the labels that should be scored for this specific crop from the test_crop_manifest
            crop_labels = get_test_crop_labels(crop.id)
            # Filter to only include labels that are in the model's classes
            filtered_classes = [c for c in classes if c in crop_labels]

            # Create the writer
            # Note: We pass all classes to the model for prediction, but only the filtered
            # classes will be saved by the CellMapDatasetWriter
            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(
                        crop=f"crop{crop.id}",
                        dataset=crop.dataset,
                    ),
                    "classes": filtered_classes,
                    "model_classes": classes,  # All classes the model was trained on
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": device,
                    "raw_value_transforms": value_transforms,
                }
            )
    else:
        crop_list = crops.split(",")
        crop_paths = []
        for i, crop in enumerate(crop_list):
            if (isinstance(crop, str) and crop.isnumeric()) or isinstance(crop, int):
                crop = f"crop{crop}"
                crop_list[i] = crop  # type: ignore

            crop_paths.extend(
                glob(
                    search_path.format(
                        dataset="*", name=crop_name.format(crop=crop, label="")
                    ).rstrip(os.path.sep)
                )
            )

        dataset_writers = []
        for crop, crop_path in zip(crop_list, crop_paths):  # type: ignore
            # Get path to raw dataset
            raw_path = get_raw_path(crop_path, label="")

            # Get the boundaries of the crop
            gt_images = {
                array_name: CellMapImage(
                    str(UPath(crop_path) / classes[0]),
                    target_class=classes[0],
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for array_name, array_info in target_arrays.items()
            }

            target_bounds = {
                array_name: image.bounding_box
                for array_name, image in gt_images.items()
            }

            dataset = get_formatted_fields(raw_path, search_path, ["{dataset}"])[
                "dataset"
            ]

            # Create the writer
            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(crop=crop, dataset=dataset),
                    "classes": classes,
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": device,
                    "raw_value_transforms": value_transforms,
                }
            )

    for dataset_writer in dataset_writers:
        predict_func(model, dataset_writer, batch_size)
