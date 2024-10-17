from glob import glob
from importlib.machinery import SourceFileLoader
import tempfile
import torch
from typing import Optional, Sequence
import os
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
import numpy as np
from upath import UPath

from cellmap_segmentation_challenge.utils.datasplit import (
    REPO_ROOT,
    SEARCH_PATH,
    RAW_NAME,
    get_csv_string,
    make_datasplit_csv,
)


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
            os.path.join(tmp_dir.name, f"output.zarr", axis),
            input_block_shape,
            channels,
            roi,
            min_raw,
            max_raw,
        )

    # Combine the predictions from the x, y, and z orthogonal planes
    raw_dataset = open_ds(in_dataset)
    for label in channels.values():
        # Load the predictions from the x, y, and z orthogonal planes
        predictions = []
        for axis in range(3):
            predictions.append(
                open_ds(os.path.join(tmp_dir.name, f"output.zarr", axis, label))[:]
            )

        # Combine the predictions
        combined_predictions = np.mean(predictions, axis=0)

        # Save the combined predictions
        example_ds = open_ds(os.path.join(tmp_dir.name, f"output.zarr", axis, label))
        total_write_roi = example_ds.roi
        dataset = prepare_ds(
            f"{out_dataset}/{label}",
            total_roi=total_write_roi,
            voxel_size=raw_dataset.voxel_size,
            write_size=example_ds.chunk_shape,
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
    min_raw : float, optional
        The minimum value of the raw data. Default is 0.
    max_raw : float, optional
        The maximum value of the raw data. Default is 255.
    """
    model.eval()

    shift = min_raw
    scale = max_raw - min_raw

    raw_dataset = open_ds(in_dataset)
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
        total_write_roi = parsed_roi.snap_to_grid(raw_dataset.voxel_size)
        total_read_roi = total_write_roi.grow(context, context)
    else:
        total_read_roi = raw_dataset.roi
        total_write_roi = total_read_roi.grow(-context, -context)

    if len(input_block_shape) == 2:
        input_block_shape = (1,) + input_block_shape

    read_shape = input_block_shape * raw_dataset.voxel_size
    write_shape = get_output_shape(model, input_block_shape)

    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)

    if isinstance(channels, Sequence):
        channels = {i: c for i, c in enumerate(channels)}

    out_datasets = {}
    for channel, label in channels.items():
        dataset = prepare_ds(
            f"{out_dataset}/{label}",
            total_roi=total_write_roi,
            voxel_size=raw_dataset.voxel_size,
            write_size=write_roi.shape,
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
    output_path: str = str(
        REPO_ROOT / "data/predictions/predictions.zarr/{crop}/{label}"
    ),
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
    classes = config.classes

    if do_orthoplanes and any([s == 1 for s in input_block_shape]):
        # If the model is a 2D model, compute the average of predictions from x, y, and z orthogonal planes
        predict_func = predict_ortho_planes
    else:
        predict_func = _predict

    # Get the crops to predict on
    if crops == "test":
        crops_paths = glob(SEARCH_PATH.format(dataset="*", label="cell"))
    else:
        ...  # TODO

    for crop in crops.split(","):

        crop_path = SEARCH_PATH.format(dataset="*", label=RAW_NAME)

        in_dataset = glob(SEARCH_PATH.format(dataset="*", label=RAW_NAME))


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
