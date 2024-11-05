from glob import glob
import os
from upath import UPath
from .utils import load_safe_config
from .utils.datasplit import REPO_ROOT, SEARCH_PATH, CROP_NAME, get_raw_path
from cellmap_data import CellMapImage, CellMapDatasetWriter

def _process(...):
    ...

def process(
    config_path: str | UPath,
    crops: str = "test",
    input_path: str = UPath(
        REPO_ROOT / "data/predictions/predictions.zarr/{crop}"
    ).path,
    output_path: str = UPath(REPO_ROOT / "data/predictions/processed.zarr/{crop}").path,
    overwrite: bool = False,
) -> None:
    """
    Process and save arrays using an arbitrary process function defined in a config python file.

    Parameters
    ----------
    config_path : str | UPath
        The path to the python file containing the process function and other configurations. The script should specify the process function as `process_func`; `input_array_info` and `target_array_info` corresponding to the chunk sizes and scales for the input and output datasets, respectively; `batch_size`; `classes`; and any other required configurations.
        The process function should take a numpy array as input and return a numpy array as output.
    crops: str, optional
        A comma-separated list of crop numbers to predict on, or "test" to predict on the entire test set. Default is "test".
    input_path: str, optional
        The path to the data to process, formatted as a string with a placeholder for the crop number. Default is "cellmap-segmentation-challenge/data/predictions/predictions.zarr/{crop}".
    output_path: str, optional
        The path to save the output predictions to, formatted as a string with a placeholders for the crop number, and label class. Default is "cellmap-segmentation-challenge/data/predictions/processed.zarr/{crop}/{label}".
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
        raw_search_label = "test"
        crop_search_label = ""
        crops_paths = glob(
            SEARCH_PATH.format(
                dataset="*", name=CROP_NAME.format(crop="*", label="test")
            ).rstrip(os.path.sep)
        )
        
        # Make crop list
        crops_list = [UPath(crop_path).parts[-2] for crop_path in crops_paths]
        crop_paths = [input_path.format(crop=crop) for crop in crops_list]
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
                    input_path.format(
                        dataset="*", name=CROP_NAME.format(crop=f"crop{crop}")
                    ).rstrip(os.path.sep)
                )
            )



    dataset_writers = []
    for crop_path in crops_paths:
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
                "target_path": output_path.format(crop=UPath(crop_path).stem),
                "classes": classes,
                "input_arrays": input_arrays,
                "target_arrays": target_arrays,
                "target_bounds": target_bounds,
                "overwrite": overwrite,
            }
        )

    for dataset_writer in dataset_writers:
        _process(...)
