# This file is used to predict the segmentation logits of the 3D test datasets using the model trained in the train_2D.py script.
# %%
# Imports
from cellmap_segmentation_challenge import predict

config_path = __file__.replace("predict", "train")

# Overwrite the predictions if they already exist
predict(config_path, crops="19", overwrite=True)

# %%
