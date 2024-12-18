Evaluation
==========

This document describes how submitted data is scored in the CellMap Segmentation Challenge.

Resampling
----------
Before scoring, the predicted volumes are resampled to ensure they are compared to the ground truth at the same resolution and region of interest (ROI). For more details on the resampling process, refer to `evaluation_resampling.rst`.

Instance Segmentations
----------------------

- **Classes Included**: The following classes are included in the instance segmentation evaluation:
  - Cell (`cell`)
  - Endosome (`endo`)
  - Lipid droplet (`ld`)
  - Lysosome (`lyso`)
  - Mitochondria (`mito`)
  - Microtubule (`mt`)
  - Nuclear pore (`np`)
  - Nucleus (`nuc`)
  - Peroxisome (`perox`)
  - Vesicle (`ves`)
  - Vimentin (`vim`)

- **Scoring Components**:
  - **Hausdorff Distance**: The Hausdorff distance is calculated in nanometers between the predicted and ground truth instance segmentations. This metric measures the maximum distance between any point on the predicted instance and its nearest point on the ground truth instance, and vice versa.
  - **Accuracy**: The accuracy is calculated as the proportion of correctly predicted instance labels to the total number of instance labels.

- **Score Normalization and Combination**:
  - The Hausdorff distance is normalized to a range of [0, 1] using the maximum distance represented by a voxel. Specifically, the normalized Hausdorff distance is 1.01^(- hausdorff distance / ||voxel_size||)
  - The combined score is calculated as the geometric mean of the accuracy and the normalized Hausdorff distance.

Semantic Segmentations
----------------------

- **All non-instance classesm are included as semantic labels**

- **Scoring Components**:
  - **Intersection over Union (IoU)**: The IoU is calculated as the intersection of the predicted and ground truth segmentations divided by their union. This metric measures the overlap between the predicted and ground truth segmentations.
  - **Dice Score**: The Dice score is calculated as twice the intersection of the predicted and ground truth segmentations divided by the sum of their volumes. This metric measures the similarity between the predicted and ground truth segmentations.

- **Score Normalization and Combination**:
  - The IoU scores are combined across all volumes to obtain the final scores, normalized by the volume occupied by the voxels to which each IoU corresponds.
