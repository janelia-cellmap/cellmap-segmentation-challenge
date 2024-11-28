import os
from glob import glob
from typing import Any, Callable

from cellmap_data import CellMapDatasetWriter, CellMapImage
from tqdm import tqdm
from upath import UPath

from .config import PREDICTIONS_PATH, PROCESSED_PATH
from .utils import load_safe_config, fetch_test_crop_manifest
from .utils.datasplit import get_formatted_fields


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
    input_path: str = PREDICTIONS_PATH,
    output_path: str = PROCESSED_PATH,
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
        The path to the data to process, formatted as a string with a placeholders for the crop number, dataset, and label. Default is PREDICTIONS_PATH set in `cellmap-segmentation/config.py`.
    output_path: str, optional
        The path to save the processed output to, formatted as a string with a placeholders for the crop number, dataset, and label. Default is PROCESSED_PATH set in `cellmap-segmentation/config.py`.
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
        test_crops = fetch_test_crop_manifest()
        crop_list = tuple(c.id for c in test_crops)
    else:
        crop_list = crops.split(",")

    crop_paths = []
    for i, crop in enumerate(crop_list):
        if (isinstance(crop, str) and crop.isnumeric()) or isinstance(crop, int):
            crop = f"crop{crop}"
            crop_list[i] = crop  # type: ignore

        crop_paths.extend(
            glob(input_path.format(dataset="*", crop=crop).rstrip(os.path.sep))
        )

    crop_dict = {}
    for crop, path in zip(crop_list, crop_paths):
        dataset = get_formatted_fields(path, input_path, ["{dataset}"])["dataset"]
        crop_dict[crop] = [
            input_path.format(
                crop=crop,
                dataset=dataset,
            ),
            output_path.format(
                crop=crop,
                dataset=dataset,
            ),
        ]

    dataset_writers = []
    for crop, (in_path, out_path) in crop_dict.items():
        for label in classes:
            class_in_path = str(UPath(in_path) / label)

            # Get the boundaries of the crop
            input_images = {
                array_name: CellMapImage(
                    class_in_path,
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
                    "raw_path": class_in_path,
                    "target_path": out_path,
                    "classes": [label],
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                }
            )

    for dataset_writer in dataset_writers:
        _process(dataset_writer, process_func, batch_size)
