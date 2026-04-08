Evaluation
==========

This document describes how submitted data is scored in the CellMap Segmentation Challenge.

Resampling
----------
Before scoring, the predicted volumes are resampled to ensure they are compared to the ground truth at the same resolution and region of interest (ROI). For more details on the resampling process, refer to `evaluation_resampling.rst`.

Evaluation Approach
-------------------

All labels — both instance ("thing") and semantic ("stuff") classes — are evaluated
using **Panoptic Quality (PQ)**, a unified metric that rewards both correct detection
and accurate segmentation.

For each crop and each label, the scorer produces four raw accumulators:

- **TP** (True Positives): matched instance/segment pairs with IoU > 0.5
- **FP** (False Positives): predicted instances/segments with no GT match
- **FN** (False Negatives): GT instances/segments with no predicted match
- **sum\_IoU**: sum of IoU values for all TP pairs

These are micro-averaged across crops to produce per-category PQ, SQ, and RQ scores,
and the final overall score is the unweighted mean PQ across all categories.

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

- **Scoring Method**:

  Predicted labels are first relabeled via connected components. GT and predicted
  instances are then matched greedily in descending-IoU order, keeping only pairs
  with IoU > 0.5. Because IoU > 0.5 guarantees at most one valid counterpart for
  each instance, this greedy matching is provably optimal. The result is a set of
  TP/FP/FN/sum\_IoU accumulators for the crop.

  If the predicted-to-GT instance ratio or the number of overlap edges exceeds
  configured limits, the crop is skipped and worst-case accumulators are returned
  (``tp=0``, ``fp=nP``, ``fn=nG``, ``sum_iou=0``).

Semantic Segmentations
----------------------

- **All non-instance classes are included as semantic labels**

- **Scoring Method**:

  Each semantic label is treated as a single binary segment per crop (one GT
  segment, one predicted segment). Their binary IoU is computed; if IoU > 0.5
  the crop counts as a TP match, otherwise as both an FP and an FN. This keeps
  semantic labels consistent with the same PQ framework used for instance labels.

Metrics
-------

Per-Crop Accumulators
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``tp``
     - True positives — matched pairs with IoU > 0.5
   * - ``fp``
     - False positives — unmatched predicted instances/segments
   * - ``fn``
     - False negatives — unmatched GT instances/segments
   * - ``sum_iou``
     - Sum of IoU values for all TP matches
   * - ``pq``
     - Per-crop Panoptic Quality = ``sum_iou / (TP + 0.5·FP + 0.5·FN)``
   * - ``sq``
     - Per-crop Segmentation Quality = ``sum_iou / TP`` (mean IoU of matched pairs; 0 when TP=0)
   * - ``rq``
     - Per-crop Recognition Quality (F1) = ``2·TP / (2·TP + FP + FN)``

Per-Category Scores (``label_scores``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accumulators are micro-averaged across all crops for each category:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``pq``
     - :math:`\frac{\text{global\_sum\_IoU}}{\text{global\_TP} + 0.5 \cdot \text{global\_FP} + 0.5 \cdot \text{global\_FN}}`
   * - ``sq``
     - :math:`\frac{\text{global\_sum\_IoU}}{\text{global\_TP}}` — 0 when global TP = 0
   * - ``rq``
     - :math:`\frac{\text{global\_TP}}{\text{global\_TP} + 0.5 \cdot \text{global\_FP} + 0.5 \cdot \text{global\_FN}}`
   * - ``tp``, ``fp``, ``fn``, ``sum_iou``
     - Globally accumulated raw values

Overall Scores
~~~~~~~~~~~~~~

The per-category PQ scores are combined as an **arithmetic mean across categories**
(not weighted by instance count or voxel volume), so each category contributes equally.

.. list-table::
   :header-rows: 1

   * - Metric
     - Description
   * - ``overall_thing_pq``
     - Arithmetic mean of ``pq`` across instance ("thing") categories
   * - ``overall_stuff_pq``
     - Arithmetic mean of ``pq`` across semantic ("stuff") categories
   * - ``overall_score``
     - Arithmetic mean of ``pq`` across **all** categories
   * - ``overall_instance_score``
     - Alias for ``overall_thing_pq``
   * - ``overall_semantic_score``
     - Alias for ``overall_stuff_pq``
