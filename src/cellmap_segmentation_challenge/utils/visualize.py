from typing import Sequence
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from cellmap_data.utils import get_image_dict


def save_result_figs(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    classes: Sequence[str],
    figures_save_path: str,
):
    """
    Save the input, output, and target images to the specified path.

    Parameters
    ----------
    inputs : torch.Tensor
        The input images.
    outputs : torch.Tensor
        The output images.
    targets : torch.Tensor
        The target images.
    classes : Sequence[str]
        The classes present in the images.
    figures_save_path : str
        The path to save the figures to.
    """
    # Make sure the save path exists
    os.makedirs(os.path.dirname(figures_save_path), exist_ok=True)

    figs = get_image_dict(inputs, outputs, targets, classes)
    for label, fig in figs.items():
        fig.savefig(figures_save_path.format(label=label))
        plt.close(fig)
