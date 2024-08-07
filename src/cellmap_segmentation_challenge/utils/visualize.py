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
    # Make sure the save path exists
    os.makedirs(os.path.dirname(figures_save_path), exist_ok=True)

    figs = get_image_dict(inputs, outputs, targets, classes)
    for label, fig in figs.items():
        fig.savefig(figures_save_path.format(label=label))
        plt.close(fig)


def get_loss_plot(losses, validation_scores, iterations_per_epoch):
    fig, ax = plt.subplots()
    epoch_steps = range(0, len(losses), iterations_per_epoch)
    average_losses = [
        np.mean(losses[i : i + iterations_per_epoch])
        for i in range(0, len(losses), iterations_per_epoch)
    ]
    ax.plot(
        epoch_steps,
        average_losses,
        label="Average Training Loss",
        color="blue",
        linewidth=1.5,
    )
    ax.plot(losses, label="Training Loss", alpha=0.5, color="gray", linewidth=0.5)
    ax.plot(
        epoch_steps[: len(validation_scores)],
        validation_scores,
        label="Validation Score",
        color="red",
        linewidth=1.5,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig
