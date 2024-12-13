.. _pytorch_dataloader_guide:

===========================
Creating a CellMap Data PyTorch Dataloader
===========================

This guide explains how to create and use PyTorch Cellmap data dataloaders with the provided ``get_dataloader`` function, and demonstrates how to integrate them into a training loop. The function leverages the ``CellMapDataLoader`` and ``CellMapDataSplit`` classes from the `cellmap_data` package.

Overview
--------

The ``get_dataloader`` function returns two PyTorch-compatible loaders—one for training and one for validation. These loaders handle data preprocessing, augmentation, and device placement. Key parameters include:

- **datasplit_path**: Path to a CSV file defining train/validation splits.
- **classes**: List of class names to segment (e.g., ``["nuc", "er"]``).
- **batch_size**: Batch size for training and validation.
- **array_info / input_array_info / target_array_info**: Dictionaries defining array shape and scale.
- **spatial_transforms**: Dictionary specifying spatial augmentations (mirror, transpose, rotate, etc.).
- **iterations_per_epoch**: Number of iterations per training epoch.
- **random_validation**: If ``True``, validation batches are randomly sampled.
- **device**: Compute device (e.g., ``"cpu"``, ``"cuda"``, or ``"mps"``).

Prerequisites
-------------

1. Install required dependencies: ``torch``, ``torchvision``, and ``cellmap_data``.
2. Prepare a datasplit CSV for ``CellMapDataSplit``.
3. Ensure your data and array configurations align with the input/target array info provided.

Example Usage
-------------

**Step 1: Define Parameters**

.. code-block:: python

   datasplit_path = "datasplit.csv"
   classes = ["nuc", "er"]
   batch_size = 8

   input_array_info = {"shape": (1, 128, 128), "scale": (8, 8, 8)}
   target_array_info = {"shape": (1, 128, 128), "scale": (8, 8, 8)}

   spatial_transforms = {
       "mirror": {"axes": {"x": 0.5, "y": 0.5}},
       "transpose": {"axes": ["x", "y"]},
       "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
   }

   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   iterations_per_epoch = 1000

**Step 2: Get the Dataloaders**

.. code-block:: python

   from cellmap_segmentation_challenge.utils import get_dataloader 

   train_loader, val_loader = get_dataloader(
       datasplit_path=datasplit_path,
       classes=classes,
       batch_size=batch_size,
       input_array_info=input_array_info,
       target_array_info=target_array_info,
       spatial_transforms=spatial_transforms,
       iterations_per_epoch=iterations_per_epoch,
       random_validation=False,
       device=device
   )

The returned ``train_loader`` and ``val_loader`` are ``CellMapDataLoader`` objects.

**Step 3: Using the Dataloaders in a Training Loop**

Below is a simplified training loop example. It assumes a model, loss function, and optimizer are defined.

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim

   # Example model, loss, and optimizer
   model = nn.Sequential(
       nn.Conv2d(1, 16, kernel_size=3, padding=1),
       nn.ReLU(),
       nn.Conv2d(16, len(classes), kernel_size=3, padding=1),
   ).to(device)

   criterion = nn.BCEWithLogitsLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.0001)

   epochs = 2
   for epoch in range(epochs):
       # Training phase
       model.train()
       train_loader.refresh()  # Refresh if supported
       for batch in train_loader.loader:
           inputs = batch["input"]
           targets = batch["output"]

           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, targets)
           loss.backward()
           optimizer.step()

           print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

       # Validation phase
       model.eval()
       val_loader.refresh()
       val_loss = 0
       val_count = 0
       with torch.no_grad():
           for batch in val_loader.loader:
               inputs = batch["input"]
               targets = batch["output"]
               outputs = model(inputs)
               batch_loss = criterion(outputs, targets)
               val_loss += batch_loss.item()
               val_count += 1

       if val_count > 0:
           avg_val_loss = val_loss / val_count
           print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

Notes
-----

- ``CellMapDataLoader`` and ``CellMapDataSplit`` are provided by ``cellmap_data``. Check their documentation for details on preparing your datasplit CSV and data structures.
- Adjust ``spatial_transforms`` as needed for 2D or 3D data.
- ``iterations_per_epoch`` defines how the training loader is sampled.
- If encountering issues, verify that you have the necessary dependencies and that your datasplit file and data paths are correct.

Troubleshooting
---------------

- **Missing Dependencies**: Install ``tensorboardX`` and ``upath`` if needed. Ensure PyTorch and CUDA are properly set up.
- **Data Loading Issues**: Confirm that the datasplit CSV points to valid data and that the array info matches your dataset dimensions.
- **Device Issues**: If CUDA is not available, the code falls back to CPU or MPS. Check that your GPU drivers are correctly installed.

Conclusion
----------

You can easily set up PyTorch dataloaders for segmentation tasks with the provided ``get_dataloader`` function and integrate them into a training workflow, handling normalization, augmentation, and device placement seamlessly.
