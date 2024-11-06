from glob import glob
import os
import tempfile
from typing import Any
from cellmap_data import CellMapDatasetWriter, CellMapImage
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm
from upath import UPath
from .models import load_best_val, load_latest
from .config import CROP_NAME, SEARCH_PATH, PREDICTIONS_PATH
from .utils.datasplit import get_raw_path, get_dataset_name
from cellmap_data.transforms.augment import (
    Normalize,
    NaNtoNum,
)
from .utils import load_safe_config


def predict_orthoplanes(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    print("Predicting orthogonal planes.")

    # Make a temporary prediction for each axis
    tmp_dir = tempfile.TemporaryDirectory()
    print(f"Temporary directory for predictions: {tmp_dir.name}")
    for axis in range(3):
        temp_kwargs = dataset_writer_kwargs.copy()
        temp_kwargs["target_path"] = os.path.join(
            tmp_dir.name, "output.zarr", str(axis)
        )
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
    for batch in tqdm(dataset_writer.loader(batch_size=batch_size)):
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

    value_transforms = T.Compose(
        [
            Normalize(),
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ],
    )

    dataset_writer = CellMapDatasetWriter(
        **dataset_writer_kwargs, raw_value_transforms=value_transforms
    )
    dataloader = dataset_writer.loader(batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Get the inputs and outputs
            inputs = batch["input"]
            outputs = model(inputs)
            outputs = {"output": model(inputs)}

            # Save the outputs
            dataset_writer[batch["idx"]] = outputs


def predict(
    config_path: str,
    crops: str = "test",
    output_path: str = PREDICTIONS_PATH,
    do_orthoplanes: bool = True,
    overwrite: bool = False,
):
    """
    Given a model configuration file and list of crop numbers, predicts the output of a model on a large dataset by splitting it into blocks and predicting each block separately.

    Parameters
    ----------
    config_path : str
        The path to the model configuration file. This can be the same as the config file used for training.
    crops: str, optional
        A comma-separated list of crop numbers to predict on, or "test" to predict on the entire test set. Default is "test".
    output_path: str, optional
        The path to save the output predictions to, formatted as a string with a placeholders for the dataset, crop number, and label. Default is PREDICTIONS_PATH set in `cellmap-segmentation/config.py`.
    do_orthoplanes: bool, optional
        Whether to compute the average of predictions from x, y, and z orthogonal planes for the full 3D volume. This is sometimes called 2.5D predictions. It expects a model that yields 2D outputs. Similarly, it expects the input shape to the model to be 2D. Default is True for 2D models.
    overwrite: bool, optional
        Whether to overwrite the output dataset if it already exists. Default is False.
    """
    config = load_safe_config(config_path)
    classes = config.classes
    batch_size = getattr(config, "batch_size", 8)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    model_name = getattr(config, "model_name", "2d_unet")
    model_to_load = getattr(config, "model_to_load", model_name)
    model = config.model
    load_model = getattr(config, "load_model", "latest")
    model_save_path = getattr(
        config, "model_save_path", UPath("checkpoints/{model_name}_{epoch}.pth").path
    )
    logs_save_path = getattr(
        config, "logs_save_path", UPath("tensorboard/{model_name}").path
    )

    # %% Check that the GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Prediction device: {device}")

    # %% Move model to device
    model = model.to(device)

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

    if do_orthoplanes and any([s == 1 for s in input_array_info["shape"]]):
        # If the model is a 2D model, compute the average of predictions from x, y, and z orthogonal planes
        predict_func = predict_orthoplanes
    else:
        predict_func = _predict

    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}
    assert (
        input_arrays is not None and target_arrays is not None
    ), "No array info provided"
    # Get the crops to predict on
    if crops == "test":
        # TODO: Could make this more general to work for any class label
        raw_search_label = "test"
        crop_search_label = ""
        crops_paths = glob(
            SEARCH_PATH.format(
                dataset="*", name=CROP_NAME.format(crop="*", label="test")
            )
        )
    else:
        crop_list = crops.split(",")
        assert all(
            [crop.isnumeric() for crop in crop_list]
        ), "Crop numbers must be numeric or `test`."
        crop_paths = []
        raw_search_label = ""
        crop_search_label = classes[0]
        for crop in crop_list:
            crop_paths.extend(
                glob(
                    SEARCH_PATH.format(
                        dataset="*", name=CROP_NAME.format(crop=f"crop{crop}", label="")
                    ).rstrip(os.path.sep)
                )
            )

    # Make crop list
    crops_dict = {UPath(crop_path).parts[-2]: crop_path for crop_path in crops_paths}

    dataset_writers = []
    for crop, crop_path in crops_dict.items():  # type: ignore
        # Get path to raw dataset
        raw_path = get_raw_path(crop_path, label=raw_search_label)

        # Get the boundaries of the crop
        gt_images = {
            array_name: CellMapImage(
                str(UPath(crop_path) / crop_search_label),
                target_class=classes[0],
                target_scale=array_info["scale"],
                target_voxel_shape=array_info["shape"],
                pad=True,
                pad_value=0,
            )
            for array_name, array_info in target_arrays.items()
        }

        target_bounds = {
            array_name: image.bounding_box for array_name, image in gt_images.items()
        }

        # Create the writer
        dataset_writers.append(
            {
                "raw_path": raw_path,
                "target_path": output_path.format(
                    crop=crop, dataset=get_dataset_name(raw_path)
                ),
                "classes": classes,
                "input_arrays": input_arrays,
                "target_arrays": target_arrays,
                "target_bounds": target_bounds,
                "overwrite": overwrite,
            }
        )

    for dataset_writer in dataset_writers:
        predict_func(model, dataset_writer, batch_size)
