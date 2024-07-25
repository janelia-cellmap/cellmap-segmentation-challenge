import os
import torch
import glob


def load_latest(search_path, model):
    checkpoint_files = glob.glob(search_path)
    if checkpoint_files:
        checkpoint_files.sort(
            key=os.path.getmtime, reverse=True
        )  # If there are checkpoints they are sorted by modification time
        newest_checkpoint = checkpoint_files[0]

        # Loads the most recent checkpoint into the model
        model.load_state_dict(torch.load(newest_checkpoint))
        print(f"Loaded latest checkpoint: {newest_checkpoint}")
