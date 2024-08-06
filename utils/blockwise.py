import torch
from typing import Sequence, Mapping
import os
from . import evaluate


def predict(
    model: torch.nn.Module,
    output_path: str | os.PathLike,
    data: Mapping[str, str],
    read_size: Sequence[int],
    write_size: Sequence[int],
    scale: Sequence[int],
    num_workers=1,
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
    """
    ...


def process(
    data: Mapping[str, str],
    function_path: str | os.PathLike,
    output_path: str | os.PathLike,
    read_size: Sequence[int],
    write_size: Sequence[int],
    num_workers=1,
) -> None:
    """
    Processes the input data in blocks and saves the output data.

    Parameters
    ----------
    data : Mapping[str, str]
        A dictionary of input data paths with the name of the input dataset as the key. The name of the input dataset will be used to save the output data.
    function_path : str | os.PathLike
        The path to the file with the function to use for processing the data. Function in the file should have the signature:
            `process_function(input_data: torch.Tensor) -> output_data: Mapping[str, torch.Tensor]`
    output_path : str | os.PathLike
        The path to save the output data to.
    read_size : Sequence[int]
        The size of the blocks to read from the input data.
    write_size : Sequence[int]
        The size of the blocks to write to the output data.
    num_workers : int
        The number of workers to use for prediction.
    """
    ...


def evaluate_batch(
    data: Mapping[str, str],
    output_path: str | os.PathLike,
    read_size: Sequence[int],
    num_workers=1,
) -> None:
    """
    Evaluates the output data in blocks.

    Parameters
    ----------
    data : Mapping[str, str]
        A dictionary of input data paths with the name of the input dataset as the key. The name of the input dataset will be used to save the output data.
    output_path : str | os.PathLike
        The path to save the output data to.
    read_size : Sequence[int]
        The size of the blocks to read from the input data.
    num_workers : int
        The number of workers to use for prediction.
    """
    ...
    evaluate_func = evaluate
    evaluate_func_path = evaluate.__file__
