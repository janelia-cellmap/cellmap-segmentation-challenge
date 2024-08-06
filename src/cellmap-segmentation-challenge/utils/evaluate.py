from typing import Mapping

import torch


def evaluate(
    data: torch.Tensor,
) -> Mapping[str, float]:
    """
    Evaluates the output data in blocks.

    Parameters
    ----------
    data : torch.Tensor
        The output data to evaluate.

    Returns
    -------
    output_data : Mapping[str, float]
        The evaluated output data.
    """
    ...
