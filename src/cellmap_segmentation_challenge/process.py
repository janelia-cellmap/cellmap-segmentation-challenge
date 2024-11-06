from glob import glob
from tqdm import tqdm
import os
from typing import Any, Callable
from upath import UPath
from .utils import load_safe_config
from .config import CROP_NAME, REPO_ROOT, SEARCH_PATH
from .utils.datasplit import get_dataset_name
from cellmap_data import CellMapImage, CellMapDatasetWriter


def _process(
    dataset_writer_kwargs: dict[str, Any], process_func: Callable, batch_size: int = 8
) -> None:
    """
    Process and save arrays using an arbitrary process function.

    Parameters
    ----------
    dataset_writer_kwargs : dict
        A dictionary containing the specifications for data loading and writing.
    process_func : Callable
        The function to apply to the input data. Should take an array as input and return an array as output.
    batch_size : int, optional
        The batch size to use for processing the data. Default is 8.
    """
    # Create the dataset writer
    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)

    # Process the data
    for batch in tqdm(dataset_writer.loader(batch_size=batch_size)):
        # Get the input data
        inputs = batch["input"]

        # Process the data
        outputs = process_func(inputs)

        # Write the data
        dataset_writer[batch["idx"]] = {"output": outputs}


def process(
    config_path: str | UPath,
    crops: str = "test",
    input_path: str = UPath(REPO_ROOT / "data/predictions/{dataset}.zarr/{crop}").path,
    output_path: str = UPath(REPO_ROOT / "data/processed/{dataset}.zarr/{crop}").path,
    overwrite: bool = False,
) -> None:
    """
    Process and save arrays using an arbitrary process function defined in a config python file.

    Parameters
    ----------
    config_path : str | UPath
        The path to the python file containing the process function and other configurations. The script should specify the process function as `process_func`; `input_array_info` and `target_array_info` corresponding to the chunk sizes and scales for the input and output datasets, respectively; `batch_size`; `classes`; and any other required configurations.
        The process function should take an array as input and return an array as output.
    crops: str, optional
        A comma-separated list of crop numbers to process, or "test" to process the entire test set. Default is "test".
    input_path: str, optional
        The path to the data to process, formatted as a string with a placeholders for the crop number and dataset. Default is "cellmap-segmentation-challenge/data/predictions/{dataset}.zarr/{crop}".
    output_path: str, optional
        The path to save the processed output to, formatted as a string with a placeholders for the crop number and dataset. Default is "cellmap-segmentation-challenge/data/processed/{dataset}.zarr/{crop}".
    overwrite: bool, optional
        Whether to overwrite the output dataset if it already exists. Default is False.
    """
    config = load_safe_config(config_path)
    process_func = config.process_func
    classes = config.classes
    batch_size = getattr(config, "batch_size", 8)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)

    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}
    assert (
        input_arrays is not None and target_arrays is not None
    ), "No array info provided"

    # Get the crops to predict on
    if crops == "test":
        # TODO: Could make this more general to work for any class label
        crops_paths = glob(
            SEARCH_PATH.format(
                dataset="*", name=CROP_NAME.format(crop="*", label="test")
            ).rstrip(os.path.sep)
        )

        # Make crop list
        crops_list = [UPath(crop_path).parts[-2] for crop_path in crops_paths]
    else:
        crop_list = crops.split(",")
        assert all(
            [crop.isnumeric() for crop in crop_list]
        ), "Crop numbers must be numeric or `test`."
        crop_paths = []
        for crop in crop_list:
            crop_paths.extend(
                glob(
                    input_path.format(
                        dataset="*", name=CROP_NAME.format(crop=f"crop{crop}")
                    ).rstrip(os.path.sep)
                )
            )
    crop_dict = {
        crop: [
            input_path.format(crop=crop, dataset=get_dataset_name(path)),
            output_path.format(crop=crop, dataset=get_dataset_name(path)),
        ]
        for crop, path in zip(crops_list, crops_paths)
    }

    dataset_writers = []
    for crop, (input_path, output_path) in crop_dict.items():
        for label in classes:
            class_input_path = str(UPath(input_path) / label)

            # Get the boundaries of the crop
            input_images = {
                array_name: CellMapImage(
                    class_input_path,
                    target_class=label,
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for array_name, array_info in target_arrays.items()
            }

            target_bounds = {
                array_name: image.bounding_box
                for array_name, image in input_images.items()
            }

            # Create the writer
            dataset_writers.append(
                {
                    "raw_path": class_input_path,
                    "target_path": output_path,
                    "classes": [label],
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                }
            )

    for dataset_writer in dataset_writers:
        _process(dataset_writer, process_func, batch_size)
