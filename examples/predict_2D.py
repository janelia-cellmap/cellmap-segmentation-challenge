# This file is used to predict the segmentation logits of the test datasets using the model trained in the train_2D.py script.
# It does so by using the 'train_2D.py' configuration file, and the 'predict' function from the cellmap_segmentation_challenge package, which loads the trained model and runs inference on the test data.
#
# NOTE: This script writes raw logits/affinities, NOT label volumes.
# After running this script, run the corresponding process_2D.py script
# to convert the raw predictions into label volumes suitable for submission.
# %%
# Imports
from cellmap_segmentation_challenge import predict

config_path = __file__.replace("predict", "train")

# For test crops, only the labels listed in the test-crop manifest will be saved.
predict(config_path, crops="test", overwrite=True)

# %%
