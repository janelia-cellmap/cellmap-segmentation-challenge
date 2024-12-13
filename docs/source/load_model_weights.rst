.. _load_pretrained_model_guide:

========================================
Loading a Pretrained Model Checkpoint
========================================

This guide demonstrates how to load pretrained model checkpoints into a PyTorch model using the provided utility functions: ``load_latest`` and ``load_best_val``. These functions help streamline the process of restoring a model's state from previously saved checkpoints, which is useful for resuming training, performing inference, or evaluating model performance after training.

Overview
--------

The provided script defines two functions:

1. **load_latest(search_path, model)**:
   - Searches for the latest (most recently modified) checkpoint that matches the specified search pattern.
   - Loads the state dictionary into the given PyTorch model.
   - Useful when you want to resume training from the most recent checkpoint.

2. **load_best_val(logs_save_path, model_save_path, model, low_is_best=True)**:
   - Reads TensorBoard logs to find the checkpoint with the best validation score.
   - Loads the corresponding state dictionary into the provided model.
   - Ideal for inference or fine-tuning from the best-performing model state according to validation metrics.

Prerequisites
-------------

- A trained model with saved checkpoints.
- TensorBoard event files for determining the best validation score (if using ``load_best_val``).
- PyTorch and associated libraries (`torch`, `numpy`, `glob`, `tensorboard`) installed.

Make sure you have:

.. code-block:: shell

   pip install torch numpy tensorboard tensorboardX upath

Additionally, ensure that:

- The checkpoint files are saved in the format expected (e.g., `.pth` files).
- The `logs_save_path` directory contains TensorBoard event files for validation scores.

Function Definitions
--------------------

**load_latest(search_path, model)**

**Parameters**:

- **search_path** (str): A file pattern or directory path (e.g., ``'checkpoints/model_*.pth'``) to search for checkpoint files.
- **model** (torch.nn.Module): The model instance to load the state dictionary into.

**Behavior**:

- Finds all checkpoint files matching `search_path`.
- Sorts them by modification time (descending) to get the latest file.
- Loads the state dictionary into `model` with `strict=False` (allowing mismatched keys if any).

**Example**:

.. code-block:: python

   import torch
   from my_model import MyModel
   from load_pretrained import load_latest

   # Initialize your model
   model = MyModel()

   # Load the latest checkpoint
   load_latest("checkpoints/*.pth", model)
   # Now 'model' contains weights from the most recently saved checkpoint.

**load_best_val(logs_save_path, model_save_path, model, low_is_best=True)**

**Parameters**:

- **logs_save_path** (str): The directory containing TensorBoard event files with validation metrics.
- **model_save_path** (str): A format string for the model checkpoints (e.g., `'checkpoints/model_{epoch}.pth'`).
- **model** (torch.nn.Module): The model to load the best validation checkpoint into.
- **low_is_best** (bool): If `True`, the lowest validation score is considered best. If `False`, the highest score is best.

**Behavior**:

- Loads the TensorBoard events from `logs_save_path`.
- Extracts validation scores and determines the epoch with the best validation performance.
- Constructs the checkpoint path using `model_save_path` and best epoch.
- Loads that checkpoint into `model`.

**Example**:

.. code-block:: python

   import torch
   from my_model import MyModel
   from load_pretrained import load_best_val

   model = MyModel()

   # Suppose 'logs' directory has TensorBoard event files and 'checkpoints/model_{epoch}.pth' are your saved checkpoints.
   # If lower validation score is better (e.g., for a loss metric), keep low_is_best=True.
   load_best_val("logs", "checkpoints/model_{epoch}.pth", model, low_is_best=True)
   # 'model' now contains weights from the epoch with the best validation score.

Tutorial: Step-by-Step
----------------------

1. **Training and Saving Checkpoints**:  
   During training, save your model checkpoints regularly, for example:

   .. code-block:: python

      torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pth")

   Also, log validation metrics to TensorBoard so that the `load_best_val` function can analyze them:

   .. code-block:: python

      from torch.utils.tensorboard import SummaryWriter

      writer = SummaryWriter("logs")
      # After computing validation_loss at the end of each epoch:
      writer.add_scalar("validation", validation_loss, epoch)

2. **Find the Latest Checkpoint**:  
   If you need to resume training from your most recent checkpoint, do:

   .. code-block:: python

      model = MyModel()
      load_latest("checkpoints/model_*.pth", model)
      # Continue training from the loaded state

3. **Find the Best Validation Checkpoint**:  
   For deployment or testing, you might want the model that performed best on validation:

   .. code-block:: python

      model = MyModel()
      load_best_val("logs", "checkpoints/model_{epoch}.pth", model, low_is_best=True)
      # Model now contains the best weights based on validation metrics.

4. **Run Inference or Fine-Tuning**:  
   With the loaded model, you can now run inference on test data or fine-tune further:

   .. code-block:: python

      model.eval()
      # inference code here

Notes
-----

- The provided ``model_save_path`` should contain a placeholder for the epoch (e.g., `"{epoch}"`), allowing the function to construct the exact checkpoint filename for the best epoch.
- If no checkpoints are found for ``load_latest``, it won't modify your model.
- If TensorBoard logs don't contain a `validation` tag, `load_best_val` will fail to find a best epoch.
- If there's a mismatch in model architecture and checkpoint keys, `strict=False` allows partial loading, but ensure that keys align where possible.

Troubleshooting
---------------

- **No Checkpoints Found**: Ensure the `search_path` or `model_save_path` pattern is correct.
- **No Validation Events**: Verify that the `validation` scalar is logged to TensorBoard.
- **Key Mismatch in Checkpoints**: The model definition must match the architecture of the checkpoint. If keys differ, consider updating the model or checkpoint keys or allow partial loading.

Conclusion
----------

By using `load_latest` and `load_best_val`, you can effortlessly restore model states, resume training, or select the optimal model for inference. These utilities integrate seamlessly into the training workflow, making it easier to manage long-running experiments and experiment with different model states.
