from glob import glob
from importlib.machinery import SourceFileLoader
import tempfile
import torch
from typing import Mapping, Optional, Sequence
import os
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
import numpy as np
from upath import UPath
import zarr

from cellmap_segmentation_challenge.utils.datasplit import (
    CROP_NAME,
    REPO_ROOT,
    SEARCH_PATH,
    RAW_NAME,
    get_raw_path,
    get_csv_string,
    make_datasplit_csv,
)


def find_level(
    group, target_scale: Sequence[float]
) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Finds the multiscale level that is closest to the target scale.

    Parameters
    ----------
    group : zarr.Group
        The group to search for the multiscale levels.
    target_scale : Sequence[float]
        The target scale to find the closest multiscale level to.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        The name of the closest multiscale level, the offset of the level, and the scale of the level.
    """
    last_path: str | None = None
    offset = None
    scale = None
    for level in group.attrs["multiscales"][0]["datasets"]:
        for transform in level["coordinateTransformations"]:
            if "translation" in transform:
                offset = np.array(transform["translation"])
            if "scale" in transform:
                scale = np.array(transform["scale"])
            if offset is not None and scale is not None:
                break
        for ts, s in zip(target_scale, scale):
            if ts <= s:
                if last_path is None:
                    return level["path"], offset, scale
                else:
                    return last_path, offset, scale
        last_path = level["path"]
    return last_path, offset, scale  # type: ignore


def get_crop_roi(crop_path: str, output_scale: Sequence[int]) -> str:
    """
    Get the ROI of a crop from the crop path, formatted as a string: [start1:end1,start2:end2,...].

    Parameters
    ----------
    crop_path : str
        The path to the crop.

    Returns
    -------
    roi_str : str
        The ROI of the crop formatted as a string.
    """
    crop_group = zarr.open(crop_path)  # type: ignore
    scale_level, offset, scale = find_level(crop_group, output_scale)
    crop_dataset = zarr.open(UPath(crop_path) / scale_level)  # type: ignore
    shape = np.array(crop_dataset.shape) * scale
    roi = f"[{','.join([str(o) for o in offset])}:{','.join([str(o + s) for o, s in zip(offset, shape)])}]"
    return roi


def get_output_shape(
    model: torch.nn.Module, input_shape: Sequence[int]
) -> Sequence[int]:
    """
    Computes the output shape of a model given an input shape.

    Parameters
    ----------
    model : torch.nn.Module
        The model to compute the output shape for.
    input_shape : Sequence[int]
        The input shape of the model.

    Returns
    -------
    Sequence[int]
        The output shape of the model.
    """
    input_tensor = torch.zeros((1, 1), *input_shape)
    output_tensor = model(input_tensor)
    return output_tensor.shape[2:]


def predict_ortho_planes(
    model: torch.nn.Module,
    in_dataset: str | os.PathLike,
    out_dataset: str | os.PathLike,
    input_block_shape: Sequence[int],
    channels: Sequence[str] | dict[str | int, str],
    roi: Optional[str] = None,
    output_voxel_size: Optional[Sequence[int]] = None,
    min_raw: float = 0,
    max_raw: float = 255,
) -> None:
    """
    Predicts the average 3D output of a 2D model on a large dataset by splitting it into blocks and predicting each block separately, then averaging the predictions from the x, y, and z orthogonal planes.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction.
    in_dataset: str | os.PathLike
        The path to the input dataset.
    out_dataset: str | os.PathLike
        The path to the output dataset, not including the name of the channel.
    input_block_shape : Sequence[int]
        The shape of the input slices to use for prediction.
    channels : Sequence[str] | dict[str | int, str]
        The label classes to predict. The output will be saved in separate datasets for each label class. If multiple output channels belong to the same label class (such as for x,y,z affinities), they can be combined by indicating chanel:label matches with a dictionary and specifying the channels as a string of the form "0-2". Example: {"0-2":"nuc"}. For single channel per label class predictions, the channels can be specified as strings in a list or tuple.
    roi : str, optional
        The region of interest to predict. If None, the entire dataset will be predicted.
        The format is a string of the form "[start1:end1,start2:end2,...]".
    output_voxel_size : Sequence[int | float], optional
        The voxel size of the output data. Default is the same as the input voxel size.
    min_raw : float, optional
        The minimum value of the raw data. Default is 0.
    max_raw : float, optional
        The maximum value of the raw data. Default is 255.
    """

    print("Predicting orthogonal planes.")

    # Make a temporary prediction for each axis
    tmp_dir = tempfile.TemporaryDirectory()
    print(f"Temporary directory for predictions: {tmp_dir.name}")
    for axis in range(3):
        _predict(
            model,
            in_dataset,
            os.path.join(tmp_dir.name, "output.zarr", str(axis)),
            input_block_shape,
            channels,
            roi,
            output_voxel_size,
            min_raw,
            max_raw,
        )

    # Combine the predictions from the x, y, and z orthogonal planes
    raw_dataset = open_ds(in_dataset)
    if output_voxel_size is None:
        output_voxel_size = raw_dataset.voxel_size
    labels = channels if isinstance(channels, Sequence) else channels.values()
    for label in labels:
        # Load the predictions from the x, y, and z orthogonal planes
        predictions = []
        for axis in range(3):
            predictions.append(
                open_ds(os.path.join(tmp_dir.name, f"output.zarr", str(axis), label))[:]
            )

        # Combine the predictions
        combined_predictions = np.mean(predictions, axis=0)

        # Save the combined predictions
        example_ds = open_ds(
            os.path.join(tmp_dir.name, f"output.zarr", str(axis), label)
        )
        dataset = prepare_ds(
            f"{out_dataset}/{label}",
            shape=example_ds.roi.shape,
            offset=example_ds.roi.offset,
            voxel_size=example_ds.voxel_size,
            chunk_shape=example_ds.chunk_shape,
            dtype=example_ds.dtype,
        )
        dataset[:] = combined_predictions

    tmp_dir.cleanup()


def _predict(
    model: torch.nn.Module,
    in_dataset: str | os.PathLike,
    out_dataset: str | os.PathLike,
    input_block_shape: Sequence[int],
    channels: Sequence[str] | dict[str | int, str],
    roi: Optional[str] = None,
    output_voxel_size: Optional[Sequence[int]] = None,
    min_raw: float = 0,
    max_raw: float = 255,
) -> None:
    """
    Predicts the output of a model on a large dataset by splitting it into blocks
    and predicting each block separately.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction.
    in_dataset: str | os.PathLike
        The path to the input dataset.
    out_dataset: str | os.PathLike
        The path to the output dataset, not including the name of the channel.
    input_block_shape : Sequence[int]
        The shape of the input blocks to use for prediction.
    channels : Sequence[str] | dict[str | int, str]
        The label classes to predict. The output will be saved in separate datasets for each label class. If multiple output channels belong to the same label class (such as for x,y,z affinities), they can be combined by indicating chanel:label matches with a dictionary and specifying the channels as a string of the form "0-2". Example: {"0-2":"nuc"}. For single channel per label class predictions, the channels can be specified as strings in a list or tuple.
    roi : str, optional
        The region of interest to predict. If None, the entire dataset will be predicted.
        The format is a string of the form "[start1:end1,start2:end2,...]".
    output_voxel_size : Sequence[int | float], optional
        The voxel size of the output data. Default is the same as the input voxel size.
    min_raw : float, optional
        The minimum value of the raw data. Default is 0.
    max_raw : float, optional
        The maximum value of the raw data. Default is 255.
    """
    model.eval()

    shift = min_raw
    scale = max_raw - min_raw

    raw_dataset = open_ds(in_dataset)
    if output_voxel_size is None:
        output_voxel_size = raw_dataset.voxel_size

    if len(input_block_shape) == 2:
        input_block_shape = (1,) + tuple(input_block_shape)

    read_shape = daisy.Coordinate(input_block_shape)
    write_shape = daisy.Coordinate(get_output_shape(model, input_block_shape))

    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)

    if roi is not None:
        parsed_start, parsed_end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        parsed_roi = daisy.Roi(
            daisy.Coordinate(parsed_start),
            daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
        )
        total_write_roi = parsed_roi.snap_to_grid(output_voxel_size)
        total_read_roi = total_write_roi.grow(context, context).snap_to_grid(
            raw_dataset.voxel_size, mode="grow"
        )
    else:
        total_read_roi = raw_dataset.roi
        total_write_roi = total_read_roi.grow(-context, -context).snap_to_grid(
            output_voxel_size
        )

    if isinstance(channels, Sequence):
        channels = {i: c for i, c in enumerate(channels)}

    out_datasets = {}
    for channel, label in channels.items():
        dataset = prepare_ds(
            f"{out_dataset}/{label}",
            shape=total_write_roi.shape,
            offset=total_write_roi.offset,
            voxel_size=raw_dataset.voxel_size,
            chunk_shape=write_roi.shape,
            dtype=np.float32,
        )
        out_datasets[channel] = dataset

    def predict_worker():
        client = daisy.Client()
        device = model.device
        while True:
            with client.acquire_block() as block:
                if block is None:
                    break

                raw_input = (
                    2.0
                    * (
                        raw_dataset.to_ndarray(
                            roi=block.read_roi, fill_value=shift + scale
                        ).astype(np.float32)
                        - shift
                    )
                    / scale
                ) - 1.0
                raw_input = np.expand_dims(raw_input, (0, 1))
                write_roi = block.write_roi  # .intersect(out_datasets[0].roi)

                with torch.no_grad():
                    predictions = Array(
                        model.forward(torch.from_numpy(raw_input).float().to(device))
                        .detach()
                        .cpu()
                        .numpy()[0],
                        block.write_roi,
                        dataset.voxel_size,
                    )

                    write_data = predictions.to_ndarray(write_roi)
                    for i, out_dataset in enumerate(out_datasets):
                        if "-" in i:
                            indexes = i.split("-")
                            indexes = np.arange(int(indexes[0]), int(indexes[1]) + 1)
                        else:
                            indexes = [int(i)]
                        if len(indexes) > 1:
                            out_dataset[write_roi] = np.stack(
                                [write_data[j] for j in indexes], axis=0
                            )
                        else:
                            out_dataset[write_roi] = write_data[indexes[0]]
                block.status = daisy.BlockStatus.SUCCESS

    task = daisy.Task(
        f"predict_{in_dataset}",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=predict_worker,
        check_function=None,
        read_write_conflict=False,
        # fit="overhang",
        num_workers=1,
        max_retries=0,
        timeout=None,
    )
    daisy.run_blockwise([task])


def predict(
    config_path: str,
    crops: str = "test",
    output_path: str = str(REPO_ROOT / "data/predictions/predictions.zarr/{crop}"),
    do_orthoplanes: bool = True,
):
    """
    Given a model configuration file and list of crop numbers, predicts the output of a model on a large dataset by splitting it into blocks
    and predicting each block separately.

    Parameters
    ----------
    config_path : str
        The path to the model configuration file. This can be the same as the config file used for training.
    crops: str, optional
        A comma-separated list of crop numbers to predict on, or "test" to predict on the entire test set. Default is "test".
    output_path: str, optional
        The path to save the output predictions to, formatted as a string with a placeholders for the crop number, and label class. Default is "cellmap-segmentation-challenge/data/predictions/predictions.zarr/{crop}/{label}".
    do_orthoplanes: bool, optional
        Whether to compute the average of predictions from x, y, and z orthogonal planes for the full 3D volume. This is sometimes called 2.5D predictions. It expects a model that yields 2D outputs. Similarly, it expects the input shape to the model to be 2D. Default is True for 2D models.
    """
    config = SourceFileLoader(UPath(config_path).stem, str(config_path)).load_module()
    model = config.model
    input_block_shape = config.input_array_info["shape"]
    input_scale = config.input_array_info["scale"]
    output_scale = config.target_array_info["scale"]
    classes = config.classes

    if do_orthoplanes and any([s == 1 for s in input_block_shape]):
        # If the model is a 2D model, compute the average of predictions from x, y, and z orthogonal planes
        predict_func = predict_ortho_planes
    else:
        predict_func = _predict

    # Get the crops to predict on
    if crops == "test":
        # TODO: Could make this more general to work for any class label
        raw_search_label = crop_search_label = "test"
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

    crop_args = []
    for crop_path in crops_paths:
        # Find raw scale level
        raw_path = get_raw_path(crop_path, label=raw_search_label)
        raw_group = zarr.open(raw_path)
        scale_level, _, _ = find_level(raw_group, input_scale)
        in_dataset = UPath(raw_path) / scale_level
        assert in_dataset.exists(), f"Input dataset {in_dataset} does not exist."

        roi = get_crop_roi(str(UPath(crop_path) / crop_search_label), output_scale)

        out_dataset = output_path.format(crop=crop, label="")

        crop_args.append(
            {
                "in_dataset": str(in_dataset),
                "roi": roi,
                "out_dataset": out_dataset,
            }
        )

    for args in crop_args:
        predict_func(
            model,
            input_block_shape=input_block_shape,
            channels=classes,
            output_voxel_size=output_scale,
            *args,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        "m",
        type=str,
        help="Path to a script that will load the model. Often this can be the path to the training script (such as with the examples).",
    )
    parser.add_argument(
        "in_dataset",
        "in",
        type=str,
        help="Full path to the input dataset. This dataset should contain the raw data to predict on. Example: `/path/to/test_raw.zarr/em/s0`",
    )
    parser.add_argument(
        "out_dataset",
        "out",
        type=str,
        help="Path to the output dataset, minus the class label name(s). For example, the output dataset should be `/path/to/outputs.zarr/dataset_1`.",
    )
    parser.add_argument(
        "input_block_shape",
        "in_shape",
        type=Sequence[int],
        help="Shape of the input blocks to use for prediction.",
    )
    parser.add_argument(
        "channels",
        "ch",
        type=Sequence[str] | dict[str | int, str],
        help="The label classes to predict and their corresponding channels. Specify multiple channels for a single label class as a string of the form '0-2'. Example: {'0-2':'nuc'}.",
    )
    parser.add_argument(
        "roi",
        type=str,
        default=None,
        help="Region of interest to predict. Default is to use the entire ROI of the input dataset. Format is a string of the form '[start1:end1,start2:end2,...]'.",
    )
    parser.add_argument(
        "min_raw",
        "min",
        type=float,
        default=0,
        help="Minimum value of the raw data. Default is 0.",
    )
    parser.add_argument(
        "max_raw",
        "max",
        type=float,
        default=255,
        help="Maximum value of the raw data. Default is 255.",
    )
    parser.add_argument(
        "do_ortho_planes",
        "ortho",
        action="store_true",
        help="Whether to compute the average of predictions from x, y, and z orthogonal planes for the full 3D volume. This is sometimes called 2.5D predictions. It expects a model that yields 2D outputs. Similarly, it expects the `input_shape` to be 2D (i.e. a sequence of 2 integers).",
    )
    args = parser.parse_args()

    model_path = args.model
    model_script = UPath(model_path).stem
    model_script = SourceFileLoader(model_script, str(model_path)).load_module()
    model = model_script.model
    in_dataset = args.in_dataset
    out_dataset = args.out_dataset
    input_block_shape = args.input_block_shape
    channels = args.channels
    roi = args.roi
    min_raw = args.min_raw
    max_raw = args.max_raw
    do_ortho_planes = args.do_ortho

    # if isinstance(channels, str):
    #     channels = {
    #         i: c for channel in channels.split(",") for i, c in channel.split(":")
    #     }
    # elif:
    #     channels = {i: c for i, c in enumerate(channels)}

    # parsed_channels = [channel.split(":") for channel in channels.split(",")]

    print(f"Predicting on dataset {in_dataset} and saving to {out_dataset}.")

    if do_ortho_planes:
        # If the model is a 2D model, compute the average of predictions from x, y, and z orthogonal planes
        if len(input_block_shape) > 2:
            raise ValueError(
                "The input shape must be 2D for computing orthogonal planes."
            )
        predict_ortho_planes(
            model,
            in_dataset,
            out_dataset,
            input_block_shape,
            channels,
            roi,
            min_raw,
            max_raw,
        )
    else:
        _predict(
            model,
            in_dataset,
            out_dataset,
            input_block_shape,
            channels,
            roi,
            min_raw,
            max_raw,
        )
