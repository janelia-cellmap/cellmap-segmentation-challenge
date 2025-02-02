# Imports
import glob
import os

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
from upath import UPath


def load_latest(search_path, model):
    """
    Load the latest checkpoint from a directory into a model (in place).

    Parameters
    ----------
    search_path : str
        The path to search for checkpoints.
    model : torch.nn.Module
        The model to load the checkpoint into.
    """

    # Check if there are any files matching the checkpoint save path
    checkpoint_files = glob.glob(search_path)
    if checkpoint_files:

        # If there are checkpoints, sort by modification time
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        # Saves the latest checkpoint
        newest_checkpoint = checkpoint_files[0]

        # Loads the most recent checkpoint into the model and prints out the file path
        try:
            model.load_state_dict(
                torch.load(newest_checkpoint, weights_only=True), strict=False
            )
            print(f"Loaded latest checkpoint: {newest_checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {newest_checkpoint}")
            print(e)


def load_best_val(logs_save_path, model_save_path, model, low_is_best=True):
    """
    Load the model weights with the best validation score from a directory into an existing model object in place.

    Parameters
    ----------
    logs_save_path : str
        The path to the directory with the tensorboard logs.
    model_save_path : str
        The path to the model checkpoints.
    model : torch.nn.Module
        The model to load the checkpoint into.
    low_is_best : bool
        Whether a lower validation score is better.
    """
    # Load the event file
    try:
        print("Loading events files, may take a minute")
        event_acc = event_accumulator.EventAccumulator(logs_save_path)
        event_acc.Reload()
    except:
        print("No events file found, skipping")
        return

    # Get validation scores
    tags = event_acc.Tags()["scalars"]
    if "validation" in tags:
        events = event_acc.Scalars("validation")
        scores = [event.value for event in events]

        # Find the best score
        if low_is_best:
            best_epoch = np.argmin(scores)
        else:
            best_epoch = np.argmax(scores)

        # Load the model with the best validation score
        checkpoint_path = UPath(model_save_path.format(epoch=best_epoch)).path
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        try:
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded best validation checkpoint from epoch: {best_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {checkpoint_path}")
            print(e)
