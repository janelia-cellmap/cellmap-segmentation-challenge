# Imports
import os
import torch
import glob


# load_latest checks to see if there are any checkpoints and loads in the latest one
def load_latest(search_path, model):
    # Check if there are any files matching the checkpoint save path
    checkpoint_files = glob.glob(search_path)
    if checkpoint_files:
        # If there are checkpoints, sort by modification time
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        # Saves the latest checkpoint
        newest_checkpoint = checkpoint_files[0]

        # Loads the most recent checkpoint into the model and prints out the file path
        model.load_state_dict(torch.load(newest_checkpoint))
        print(f"Loaded latest checkpoint: {newest_checkpoint}")
