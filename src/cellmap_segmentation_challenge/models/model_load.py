# Imports
from glob import glob
import os

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
from upath import UPath
from cellmap_segmentation_challenge.utils import get_formatted_fields, format_string


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
    checkpoint_files = glob(format_string(search_path, {"epoch": "*"}))
    if checkpoint_files:

        # If there are checkpoints, sort by modification time and get the latest
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        # Get the latest checkpoint
        newest_checkpoint = checkpoint_files[0]

        # Extract the epoch from the filename
        epoch = int(
            get_formatted_fields(newest_checkpoint, search_path, ["{epoch}"])["epoch"]
        )

        # Loads the most recent checkpoint into the model and prints out the file path
        try:
            model.load_state_dict(
                torch.load(newest_checkpoint, weights_only=True), strict=False
            )
            print(f"Loaded latest checkpoint: {newest_checkpoint}")
            return epoch
        except Exception as e:
            print(f"Error loading checkpoint: {newest_checkpoint}")
            print(e)

    # If there are no checkpoints, or an error occurs, return None
    return None


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
    except FileNotFoundError:
        print("No events file found, skipping")
        return None

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
            return best_epoch
        except Exception as e:
            print(f"Error loading checkpoint: {checkpoint_path}")
            print(e)
            return None
    else:
        print("No validation scores found, skipping")
        return None


def get_best_val_epoch(logs_save_path, low_is_best=True):
    """
    Get the epoch with the best validation score from tensorboard logs.

    Parameters
    ----------
    logs_save_path : str
        The path to the directory with the tensorboard logs.
    low_is_best : bool
        Whether a lower validation score is better.

    Returns
    -------
    int or None
        The epoch number with the best validation score, or None if not found.
    """
    try:
        event_acc = event_accumulator.EventAccumulator(logs_save_path)
        event_acc.Reload()
    except:
        print("No events file found, skipping")
        return None

    tags = event_acc.Tags()["scalars"]
    if "validation" in tags:
        events = event_acc.Scalars("validation")
        scores = [event.value for event in events]

        if low_is_best:
            best_epoch = np.argmin(scores)
        else:
            best_epoch = np.argmax(scores)

        return best_epoch
    else:
        print("No validation scores found, skipping")
        return None


def get_latest_checkpoint_epoch(model_save_path):
    """
    Get the latest checkpoint epoch from a directory.

    Parameters
    ----------
    model_save_path : str
        The path to the directory with the model checkpoints.

    Returns
    -------
    int or None
        The epoch number of the latest checkpoint, or None if not found.
    """
    # Check if there are any files matching the checkpoint save path
    checkpoint_files = glob(format_string(model_save_path, {"epoch": "*"}))
    if checkpoint_files:
        # If there are checkpoints, sort by modification time and get the latest
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        # Get the latest checkpoint
        newest_checkpoint = checkpoint_files[0]

        # Extract the epoch from the filename
        epoch = int(
            get_formatted_fields(newest_checkpoint, model_save_path, ["{epoch}"])[
                "epoch"
            ]
        )
        return epoch

    # If there are no checkpoints, return None
    return None


def newest_wildcard_path(search_path):
    """
    Get the newest file matching a wildcard search path.

    Parameters
    ----------
    search_path : str
        The path to search for files.

    Returns
    -------
    str or None
        The path to the newest file, or None if no files are found.
    """
    # Check if there are any files matching the search path
    files = glob(search_path)
    if files:
        # Sort by modification time and get the latest
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    # If no files are found, return None
    return None
