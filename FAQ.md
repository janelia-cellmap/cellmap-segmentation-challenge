

# Frequently Asked Questions
---
#### This FAQ is a living document and will continue to evolve as questions and needs arise. If you have a question that isn't addressed here, please feel free to open an issue in the repository. We review incoming questions regularly and will move commonly asked or broadly helpful topics into the Discussions section to benefit the community. 
---
- [How can I access the ground truth data for the CellMap Segmentation Challenge?](#how-can-i-access-the-ground-truth-data-for-the-cellmap-segmentation-challenge)
- [Where can I find the `train_crop_manifest.csv` file in the CellMap Segmentation Challenge dataset?](#where-can-i-find-the-train_crop_manifestcsv-file-in-the-cellmap-segmentation-challenge-dataset)
- [Where can I find information about the annotated blocks for each dataset?](#where-can-i-find-information-about-the-annotated-blocks-for-each-dataset)
- [Is it normal to encounter missing chunks in the datasets after downloading them using the default `csc fetch-data` command?](#is-it-normal-to-encounter-missing-chunks-in-the-datasets-after-downloading-them-using-the-default-csc-fetch-data-command)
- [How can I download additional raw data beyond the annotated regions?](#how-can-i-download-additional-raw-data-beyond-the-annotated-regions)
- [Why do the ground truth annotations in certain crops contain category IDs that differ from the required prediction categories?](#why-do-the-ground-truth-annotations-in-certain-crops-contain-category-ids-that-differ-from-the-required-prediction-categories)
- [What does the category ID '0' represent in the ground truth annotations?](#what-does-the-category-id-0-represent-in-the-ground-truth-annotations)
- [If I encounter class '2' in the ground truth, should I map it to '60' or keep it as '2'?](#if-i-encounter-class-2-in-the-ground-truth-should-i-map-it-to-60-or-keep-it-as-2)
- [In the all groundtruth files, does each unique index correspond to the same category across different datasets?](#in-the-all-groundtruth-files-does-each-unique-index-correspond-to-the-same-category-across-different-datasets)
- [Why are classes 56–58 missing from the annotation classes in the CellMap Segmentation Challenge?](#why-are-classes-5658-missing-from-the-annotation-classes-in-the-cellmap-segmentation-challenge)
- [How can I define training hyperparameters in the CellMap Segmentation Challenge?](#how-can-i-define-training-hyperparameters-in-the-cellmap-segmentation-challenge)
- [What should I do if I encounter a `"bad_malloc"` error while running `train_2D.py`?](#what-should-i-do-if-i-encounter-a-bad_malloc-error-while-running-train_2dpy)
- [Why do I receive an `"AssertionError: No valid training datasets found"` when setting `classes = ["ribo"]` in the training script?](#why-do-i-receive-an-assertionerror-no-valid-training-datasets-found-when-setting-classes--ribo-in-the-training-script)
- [Does the dataloader automatically exclude nuclei samples with 32nm resolution?](#does-the-dataloader-automatically-exclude-nuclei-samples-with-32nm-resolution)
- [Is there a way to filter out empty data when generating a datasplit?](#is-there-a-way-to-filter-out-empty-data-when-generating-a-datasplit)
- [Why do some input arrays contain only zeros in the CellMap Segmentation Challenge dataset?](#why-do-some-input-arrays-contain-only-zeros-in-the-cellmap-segmentation-challenge-dataset)
- [I encountered some crops, like `jrc_hela-3/crop60`, where the groundtruth is all zeros, and the original image appears as random noise. Is this expected?](#i-encountered-some-crops-like-jrc_hela-3crop60-where-the-groundtruth-is-all-zeros-and-the-original-image-appears-as-random-noise-is-this-expected)
- [Why do different data chunks (e.g., jrc_mus-kidney-3, jrc_mus-liver-zon-1) have varying numbers of categories in their groundtruth folders?](#why-do-different-data-chunks-eg-jrc_mus-kidney-3-jrc_mus-liver-zon-1-have-varying-numbers-of-categories-in-their-groundtruth-folders)
- [Is the highest resolution level (`s0`) consistently 2nm across all datasets in the CellMap Segmentation Challenge?](#is-the-highest-resolution-level-s0-consistently-2nm-across-all-datasets-in-the-cellmap-segmentation-challenge)
- [Do different organelle labels have different resolutions?](#do-different-organelle-labels-have-different-resolutions)
- [When fetching data without `--fetch-all-em-resolutions`, does each crop contain different resolution data based on the labels present?](#when-fetching-data-without---fetch-all-em-resolutions-does-each-crop-contain-different-resolution-data-based-on-the-labels-present)
- [If a crop contains 8nm resolution EM data, does it automatically include lower resolutions like 16nm or 32nm?](#if-a-crop-contains-8nm-resolution-em-data-does-it-automatically-include-lower-resolutions-like-16nm-or-32nm)
- [Is there a limit to the number of submissions I can make per day in the CellMap Segmentation Challenge?](#is-there-a-limit-to-the-number-of-submissions-i-can-make-per-day-in-the-cellmap-segmentation-challenge)
- [How long does it take to receive evaluation scores after submitting a prediction result?](#how-long-does-it-take-to-receive-evaluation-scores-after-submitting-a-prediction-result)
- [Is there a way to crop only the image regions with ground truth labels and ignore the zero-only regions?](#is-there-a-way-to-crop-only-the-image-regions-with-ground-truth-labels-and-ignore-the-zero-only-regions)
- [Is the downloaded data intended for semantic segmentation, and how can I obtain labels for instance segmentation?](#is-the-downloaded-data-intended-for-semantic-segmentation-and-how-can-i-obtain-labels-for-instance-segmentation)

<br><br>


### How can I access the ground truth data for the CellMap Segmentation Challenge?

The training data for the CellMap Segmentation Challenge is hosted on the [CellMap s3 bucket](https://open.quiltdata.com/b/janelia-cosem-datasets) alongside other datasets. To download the challenge data we provide a command-line interface as part of this repo. Instructions for downloading can be found [in the README](https://github.com/janelia-cellmap/cellmap-segmentation-challenge?tab=readme-ov-file#download-the-data).

Once downloaded, the ground truth data for the CellMap Segmentation Challenge will be available in the `data` folder within the repository (unless configured otherwise). Each dataset has a corresponding subfolder that contains a [zarr store](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html) with all associated arrays. The `groundtruth` [group](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#groups) for each crop contains [multiscale groups](https://ngff.openmicroscopy.org/latest/#multiscale-md) with the annotations for each single organelle. For example, for mitochondria in crop #124 from jrc_mus-liver, you can find the ground truth data in `data/jrc_mus-liver/jrc_mus-liver.zarr/recon-1/labels/groundtruth/crop124/mito`.

<br><br>

### Where can I find the `train_crop_manifest.csv` file in the CellMap Segmentation Challenge dataset?
The `train_crop_manifest.csv` file, which provides details about the labeled training crops and their corresponding raw FIB-SEM images, is now available. You can access it directly from the challenge repository.

> **Download Link:** [train_crop_manifest.csv](https://github.com/user-attachments/files/18716486/train_crop_manifest.csv)

This manifest includes essential information such as `voxel_size`, `translation`, and `shape` for each labeled crop, aiding in accurate data alignment and analysis.

> **Note:** For further details and updates, refer to the [discussion thread](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/94) in the challenge repository.

<br><br>

### Where can I find information about the annotated blocks for each dataset?
You can refer to [the annotated blocks documentation](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/available_data.html) for details on the specific regions that have been annotated in each dataset.

<br><br>

### Is it normal to encounter missing chunks in the datasets after downloading them using the default `csc fetch-data` command?
Yes, this is expected behavior. By default, the `csc fetch-data` command downloads only the raw data blocks that have corresponding ground truth annotations. This approach optimizes storage by fetching only the annotated regions. Consequently, when viewing datasets like `jrc_cos7-1a` at the `s2` level in Fiji, you might notice missing chunks or gaps in areas without annotations.

<br><br>

### How can I download additional raw data beyond the annotated regions?
To fetch more raw data surrounding the annotated blocks, you can use the `--raw-padding` parameter with the `csc fetch-data` command. For example, to add a padding of 128 voxels:

```bash
csc fetch-data --raw-padding 128
```

Be cautious when increasing the padding size, as it may result in downloading the entire raw dataset, which could require substantial storage space.


<br><br>

### Why do the ground truth annotations in certain crops contain category IDs that differ from the required prediction categories?
The discrepancy arises because the provided ground truth annotations include a broader set of category IDs, while the challenge focuses on a specific subset for prediction. Participants should map the ground truth categories to the required prediction categories as specified in the [challenge guidelines](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/annotation_classes.html).

<br><br>

### What does the category ID '0' represent in the ground truth annotations?
In the ground truth annotations, the category ID `0` represents the background or regions without any labeled structures. Participants should ensure that their models distinguish between background (ID `0`) and the relevant organelle categories.

<br><br>

### If I encounter class '2' in the ground truth `all` array, should I map it to '60' or keep it as '2'?
Participants should refer to the challenge's [category mapping guidelines](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/annotation_classes.html#detailed-class-descriptions) to determine the correct mapping for each class. It's essential to follow the provided mappings to ensure consistency in predictions.

The `all` array includes ids at the most granular level, i.e. it will only include ids of group classes (`60`/`cell`) when the atomic classes that comprise it were not annotated. However, all annotated group and atomic classes are provided as individual arrays (in this case `2` as `pm` and `60` as `cell`). Participants can avoid resolving the group class association by using those arrays directly.

> **Note:** For detailed discussions and clarifications on annotation ID issues, please refer to [this discussion thread](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/101).

<br><br>

### In the all groundtruth files, does each unique index correspond to the same category across different datasets?
Yes, in the all groundtruth files, each unique index consistently represents the same category across different datasets.

<br><br>

### Why are classes 56–58 missing from the annotation classes in the CellMap Segmentation Challenge?
Classes 56–58 correspond to components of insulin secretory granules and are not included in the challenge's annotation classes:
- Class 56: `isg_mem` (membrane)
- Class 57: `isg_lum` (lumen)
- Class 58: `isg_ins` (insulin)

These classes are excluded because they are not part of the challenge's focus.

> **Note:** For a comprehensive list of annotation classes and their descriptions, please refer to [the project's documentation](https://janelia-cellmap.github.io/cellmap-segmentation-challenge/annotation_classes.html#detailed-class-descriptions).


<br><br>

### How can I define training hyperparameters in the CellMap Segmentation Challenge?
The training pipeline accepts a `config_path` argument that points to a Python configuration file. This file allows you to set various hyperparameters and configurations for model training. The exhaustive list of configurable parameters can be found [here in the examples README](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/examples/README.md#extended-training-configuration). By customizing these parameters in your configuration file, you can tailor the training process to your specific requirements. We encourage discussion of training strategies in the repository's [Discussion section](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions)!

<br><br>

### What should I do if I encounter a `"bad_malloc"` error while running `train_2D.py`?
The `"bad_malloc"` error typically indicates that the system has run out of GPU memory during the training process. To address this issue, consider the following steps:
- Reduce the Batch Size
- Use Smaller Input Sizes
- Optimize Model Architecture
- Upgrade GPU Hardware

> **Note:** These suggestions are general guidelines to mitigate memory allocation errors during training.

<br><br>

### Why do I receive an `"AssertionError: No valid training datasets found"` when setting `classes = ["ribo"]` in the training script?
The `ribo` class is included as a potential category; however, in the current dataset, there are no annotations for ribosomes. Consequently, datasets for the `ribo` class are empty, containing only background. This absence leads to the `"No valid training datasets found"` assertion error during training.

> **Note:** For more details, refer to [this discussion thread](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/issues/112) in the challenge repository.

<br><br>

### Does the dataloader automatically exclude nuclei samples with 32nm resolution?
No. The dataloader does not automatically exclude nuclei samples at 32nm resolution. Filtering must be manually configured in the `datasplit` settings.

<br><br>

### Is there a way to filter out empty data when generating a datasplit?
Yes. We introduced filtering to exclude empty data when creating a datasplit. Run:

```bash
csc make-datasplit -c mito,nuc -s 6.0 6.0 6.0
```

This ensures that the generated `datasplit.csv` only includes FIB-SEM and ground truth data with valid annotations.

> **Note:** Full details are available in [this discussion thread](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/128).


<br><br>

### Why do some input arrays contain only zeros in the CellMap Segmentation Challenge dataset?
This is expected. Many crops do not contain all organelles, resulting in input arrays filled with zeros or `False` values. This occurs because not every crop includes every organelle type.

<br><br>

### I encountered some crops, like `jrc_hela-3/crop60`, where the groundtruth is all zeros, and the original image appears as random noise. Is this expected?
Yes, this is expected. Some crops may not contain any annotated structures, resulting in groundtruth files filled with zeros. Additionally, certain regions might appear as noise due to the imaging process or the specific area captured. For instance, crops that specifically label portions of empty resin (non-sample) can be useful in training.

> **Note:** For more detailed information, you can refer to [this discussion thread](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/100).

<br><br>

### Why do different data chunks (e.g., jrc_mus-kidney-3, jrc_mus-liver-zon-1) have varying numbers of categories in their groundtruth folders?
Each dataset focuses on specific organelles or structures, so the number of categories in the groundtruth folders can vary accordingly.

<br><br>


### Is the highest resolution level (`s0`) consistently 2nm across all datasets in the CellMap Segmentation Challenge?
No, the highest resolution level (`s0`) varies between datasets for electron microscopy (EM) and between crops for ground truth.

> **Note:** It's essential to verify the specific resolution details for each dataset or crop when working on the challenge to ensure accurate analysis and model training.

<br><br>

### Do different organelle labels have different resolutions?
Yes. Organelle labels can appear at different resolutions. Nuclei may be labeled at 8nm or 32nm, while other organelles, like mitochondria, may have higher-resolution annotations. Larger organelles are more commonly labeled at lower resolutions, while smaller organelles tend to have higher-resolution labels.

> **Note:** More details on resolution levels can be found in [this discussion](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/93).

<br><br>

### When fetching data without `--fetch-all-em-resolutions`, does each crop contain different resolution data based on the labels present?
Yes. The EM data resolution varies depending on the organelles in each crop. If a crop contains only nuclei, it may have only 32nm EM data, while a crop with mitochondria may have 8nm EM data. This ensures that the data aligns with the labeled structures in each crop.

<br><br>

### If a crop contains 8nm resolution EM data, does it automatically include lower resolutions like 16nm or 32nm?
Yes. Lower resolutions are always included in the dataset. If 8nm resolution EM data is available, the 16nm and 32nm versions can be derived from it.

> **Note:** See [this discussion](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/100) for more details on dataset structure.

<br><br>


### Is there a limit to the number of submissions I can make per day in the CellMap Segmentation Challenge?
Currently, there is no daily submission limit for participants. However, the organizers may implement a daily or lifetime limit if the overall submission volume becomes too high or if there is suspicion of misuse of the scoring system to enhance entries.

<br><br>

### How long does it take to receive evaluation scores after submitting a prediction result?
Evaluations can take up to 3 hours to complete. The duration primarily depends on the number of unique objects within the volumes being evaluated as instance segmentations.

> **Note:** If you encounter any challenges or have further questions, feel free to ask for more in-depth assistance to troubleshoot your entry.

<br><br>


### Is there a way to crop only the image regions with ground truth labels and ignore the zero-only regions?
If you use the included data loading utilities, only regions with data should be loaded. Otherwise, you can use the `training_crop_manifest` to export cropped versions of the raw data that exclude empty regions.

<br><br>

### Is the downloaded data intended for semantic segmentation, and how can I obtain labels for instance segmentation?
The provided groundtruth data is primarily for semantic segmentation. To derive instance segmentation labels, you can apply connected component labeling or watershed algorithms to the semantic segmentation masks.

<br><br>

### Why do some classes have labels containing only 0 and 1, while others have labels containing 0,1,2... (excluding the "all" class)?
Semantic segmentation class (e.g. `ecs`) crops contain only 0's and 1's (present and not present, respectively), while classes like `cell`, which are scored for instance segmentations, will have unique object ID's (e.g. 0, 1, 2, ...). The classes scored for instance segmentations are the following:
- `nuc`
- `vim`
- `ves`
- `endo`
- `lyso`
- `ld`
- `perox`
- `mito`
- `np`
- `mt`
- `cell`
