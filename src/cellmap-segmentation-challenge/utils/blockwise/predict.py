import torch
from typing import Sequence, Mapping
import os
from enum import Enum
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
import numpy as np


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
    input_tensor = torch.zeros(*input_shape)
    output_tensor = model(input_tensor)
    return output_tensor.shape


def predict(
    model: torch.nn.Module,
    in_container: str | os.PathLike,
    in_dataset: str,
    input_block_shape: Sequence[int],
    out_container: str | os.PathLike,
    out_dataset: str,
    roi: str = None,
    channels: str = None,
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
    output_path : str | os.PathLike
        The path to save the output data to.
    data : Mapping[str, str]
        A dictionary of input data paths with the name of the input dataset as the key. The name of the input dataset will be used to save the output data.
    read_size : Sequence[int]
        The size of the blocks to read from the input data.
    write_size : Sequence[int]
        The size of the blocks to write to the output data.
    scale : Sequence[int]
        The scale of the input data in nanometers.
    num_workers : int
        The number of workers to use for prediction.
    roi : str
        The region of interest to predict on. [start1:end1,start2:end2,...]
    channels : str
        The channels to use for prediction. index:channel_name,index:channel_name,...
    """

    shift = min_raw
    scale = max_raw - min_raw

    raw_dataset = open_ds(in_container, in_dataset)
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
    else:
        parsed_roi = raw.roi

    if channels is not None:
        parsed_channels = [channel.split(":") for channel in channels.split(",")]

    read_shape = input_block_shape * raw_dataset.voxel_size
    write_shape = get_output_shape(model, input_block_shape)

    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)

    total_write_roi = parsed_roi.snap_to_grid(raw_dataset.voxel_size)
    total_read_roi = total_write_roi.grow(context, context)

    out_datasets = []

    if channels is None:
        dataset = prepare_ds(
            out_container,
            out_dataset,
            total_roi=total_write_roi,
            voxel_size=raw_dataset.voxel_size,
            write_size=write_roi.shape,
            dtype=np.float32,
        )
        out_datasets.append(dataset)
    else:
        for indexes, channel in parsed_channels:
            dataset = prepare_ds(
                out_container,
                f"{out_dataset}/{channel}",
                total_roi=total_write_roi,
                voxel_size=raw_dataset.voxel_size,
                write_size=write_roi.shape,
                dtype=np.float32,
            )
            out_datasets.append(dataset)

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
                write_roi = block.write_roi.intersect(out_datasets[0].roi)

                with torch.no_grad():
                    predictions = Array(
                        model.forward(torch.from_numpy(raw_input).float().to(device))
                        .detach()
                        .cpu()
                        .numpy()[0],
                        block.write_roi,
                        dataset.voxel_size,
                    )

                    write_data = predictions.to_ndarray(write_roi).clip(-1, 1)
                    write_data = (write_data + 1) * 255.0 / 2.0
                    for (i, _), out_dataset in zip(parsed_channels, out_datasets):
                        indexes = []
                        if "-" in i:
                            indexes = [int(j) for j in i.split("-")]
                        else:
                            indexes = [int(i)]
                        if len(indexes) > 1:
                            out_dataset[write_roi] = np.stack(
                                [write_data[j] for j in indexes], axis=0
                            ).astype(np.uint8)
                        else:
                            out_dataset[write_roi] = write_data[indexes[0]].astype(
                                np.uint8
                            )
                block.status = daisy.BlockStatus.SUCCESS
        model.eval()
        with torch.no_grad():
            raw_data = raw_dataset.to_ndarray(roi=block.read_roi, fill_value=0).astype(
                np.float32
            )
            raw_data = raw_data[np.newaxis, np.newaxis]
            raw_tensor = torch.from_numpy(raw_data)
            raw_tensor = raw_tensor.to(torch.float32)
            raw_tensor = raw_tensor.to("cuda")
            output = model(raw_tensor)
            output = output.cpu().numpy()
            out_dataset.write(
                output[0, 0],
                block.write_roi.get_offset(),
                block.write_roi.get_shape(),
            )

    task = daisy.Task(
        f"predict_{in_dataset}",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=predict_worker,
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        num_workers=1,
        max_retries=0,
        timeout=None,
    )
    daisy.run_blockwise([task])
