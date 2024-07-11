from typing import Sequence
import torch
import os
import matplotlib.pyplot as plt
from cellmap_data.utils import get_image_dict


def save_result_figs(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    classes: Sequence[str],
    figures_save_path: str,
):
    # Make sure the save path exists
    os.makedirs(os.path.dirname(figures_save_path), exist_ok=True)

    figs = get_image_dict(inputs, outputs, targets, classes)
    for label, fig in figs.items():
        fig.savefig(figures_save_path.format(label=label))
        plt.close(fig)
