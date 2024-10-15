# Outline

This document will explain the conventions for file layouts and metadata used for imaging data produced by the COSEM / Cellmap project team. Some historical background will be provided as needed to explain certain design decisions.

## COSEM

[COSEM](https://www.janelia.org/project-team/cosem) (*C*ellular *O*rganelle *S*egmentation in *E*lection *M*icroscopy) is the name of a project team that ran at [Janelia Research Campus](https://www.janelia.org/) for several years starting in 2019. The goal of this project team was use machine learning to generate complete segmentations of cellular organelles in electron microscopy images, and to distribute the resulting raw data, methods, and results widely.

This effort was successful, resulting in [several papers](https://scholar.google.com/scholar?&as_sdt=0%2C5&q=janelia+COSEM&btnG=) and [OpenOrganelle](https://www.openorganelle.org/), a web-based data portal for browsing our imaging datasets. OpenOrganelle functions as a front-end for large imaging datasets that are (as of this writing) stored on [Amazon Web Services](https://aws.amazon.com/) [S3](https://aws.amazon.com/s3/) cloud storage platform with costs covered by the generosity of Amazon's [Open Data program](https://aws.amazon.com/opendata/). The primary S3 bucket we use for hosting datasets is `s3://janelia-cosem-datasets`, which you can browse [here](https://open.quiltdata.com/b/janelia-cosem-datasets/) via a third-party bucket browsing tool.

Because the strategy and technology developed by COSEM worked well, COSEM metamorphosed into a larger project team named [Cellmap](https://www.janelia.org/project-team/cellmap).  COSEM no longer exists as a projct team, but the COSEM name is still in use for projects that were started during the COSEM era (e.g., our s3 bucket). While we could in principle rename all of these resources from COSEM to Cellmap, we deemed that this would be needlessly disruptive to anyone currently linking to our data. We did however migrate our main Github organization from [`janelia-cosem`](https://github.com/janelia-cosem/) to [`janelia-cellmap`](https://github.com/janelia-cellmap/), because Github provides a redirect link for every migrated repository, which makes the transition smooth.

So, in short, COSEM became Cellmap, but the COSEM name is still around, e.g. for old tools and resources. New tools and resources will generally use the name Cellmap.

# S3 layout 

As noted in the previous section, Cellmap stores public datasets in a single S3 bucket named `janelia-cosem-datasets`. The following sections will explain the general structure of that S3 bucket, and the motivation for the decisions that led us to adopt this design.

## What is a dataset

For the purposes of the S3 storage backend, a "dataset" is 
a collection of images, potentially from multiple modalities or analysis methods, that
all depict the same physical specimen. For example, if a tissue sample was imaged with fluorescence microscopy, followed by FIB-SEM imaging, and the resulting FIB-SEM data was used for generating segmentations, then the set `{fluorescence microscopy images, FIB-SEM images, segmentation images}` would comprise a "dataset". 

### Dataset names

By convention, datasets have structured names that start with an acronym or abbreviated form of the name of the institute that collected the data, followed by an underscore, followed by a human-readable, immutable, unique identifier for the dataset that may include some information about the sample or tissue that was imaged. We adopted this convention for a simple reason: we needed a standard name for public-facing datasets, and there wasn't a better alternative available at the time.

Note that Cellmap often publishes datasets from collaborators, and these datasets may already have names that were used internally, but if an original name does not comply with this structure, then we coin a new name that does, and we try to keep track of the old name. The same is true for the individual images within the dataset.

Some explained examples:

- `jrc_hela-2` is the name of a dataset that contains images of [HeLa cells](https://en.wikipedia.org/wiki/HeLa) imaged at Janelia Research Campus (jrc). The "2" at the end of the name denotes that this dataset is the second `jrc_hela` dataset in a sequence. 

- `jrc_mus-pacinian-corpuscle` is the name of a dataset that contains images of mouse [Pacinian corpuscle](https://en.wikipedia.org/wiki/Pacinian_corpuscle) acquired at Janelia Research Campus (jrc). This dataset has no number suffix because it is not part of a sequence. If that were to change, i.e. if another mouse Pacinian corpuscle was imaged, then the name of this dataset would stay unchanged, and the subsequent dataset would be called `jrc_mus-pacininan-corpuscle-2`.

## Datasets on S3

The top level of the `s3://janelia-cosem-datasets` bucket contains a list of prefixes (directories, in the parlance of local file systems), where each prefix is the name of a dataset. `s3://janelia-cosem-datasets/jrc_hela-2` is a URL to such a prefix. All of the images in the `jrc_hela-2` dataset are located under this prefix. 

The following objects / prefixes can be found under `s3://janelia-cosem-datasets/<dataset>/`:

- `thumbnail.jpg`: A thumbnail image for the dataset, often a screenshot taken of Neuroglancer or some other visulization tool displaying a selection of images from the dataset. This file is used by OpenOrganelle. Every dataset should have a `thumbnail.jpg` file.

- `<dataset>.n5/`: This prefix is the root of the N5 hierarchy for images that comprise the dataset which are stored in the N5 format. As of this writing, every dataset should have an N5 hierarchy root. However, we intend to migrate from N5 to Zarr, and thus newer datasets may not have an N5 hierachy. Instead, these datasets would have a prefix called `<dataset>.zarr`. For continuity, datasets that have been migrated from N5-based storage to Zarr-based storage will keep the original `<dataset>.n5` hierarchy until we feel it is safe to remove it.

- `neuroglancer/em/`: This prefix contains FIB-SEM images stored in the [Neuroglancer "precomputed" volume format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md). We used this format for a handful of datasets that were created early in the history of COSEM. At the time, the Neuroglancer "precomputed" format was chosen because it supports lossy JPEG compression, resulting in a smaller storage footprint. But use of the format never extended beyond a few older datasets, e.g., [`jrc_hela-2`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_hela-2/neuroglancer/em/). To be consistent with the `.n5` and `.zarr` suffix used by the N5 and Zarr formats respectively, images stored in the Neuroglancer precomputed format are stored under a prefix using the `.precomputed` suffix (e.g., [this example](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_hela-2/neuroglancer/em/fibsem-uint8.precomputed/))

- `neuroglancer/mesh/`: This prefix contains meshes stored in the [Neuroglancer "precomputed" mesh format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md) (e.g., [this example](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-liver/neuroglancer/mesh/mito_seg/)). Meshes are generated from segmentation images, and by convention we give the mesh the same name as the image it came from. So in the above example, the mesh data is stored under the prefix `<dataset>/neuroglancer/mesh/mito_seg/`, because it came from an [image](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-liver/jrc_mus-liver.n5/labels/mito_seg/) stored under the prefix `<dataset>/<dataset>.n5/labels/mito_seg/`. The coicindent naming is merely a convention, and could change at any time.

The force shaping this layout is the need to partition resources according to the tool that will create or consume them. All of the resources stored under the `neuroglancer/` prefix are specific to Neuroglancer; the data stored in the `<dataset>.n5/` prefix is specific to tools that understand the N5 format, etc. If a new tool comes along, we would likely introduce a new prefix with that tool's name, and store resources specific to that tool under that prefix.

## Image layout

### Chunked array format

We currently use the [`N5` format](https://github.com/saalfeldlab/n5) to store the bulk of the imaging data on S3. This format was convenient for a variety of reasons:

- We were already using N5 internally before we decided to publish data to S3.

- N5 divides arrays into chunks, where each chunk is a separate object in storage. Thus it supports massively parallel reading and writing of arrays. This is essential for reading and writing multi-TB imaging datasets.

- Neuroglancer can display images stored in N5.

Within an N5 hierarchy (i.e., under the prefix `<dataset>.n5/`), we aim to partition different images according to the method of their production. 

- Electron microscopy images are stored under a prefix with the scheme: `<dataset>.n5/em/<em modality>-<numpy-style data type specifier>/`, e.g. [`jrc_hela-2.n5/em/fibsem-uint16/`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/), which is 16 bit FIB-SEM data. We include the data type in the name of these datasets to ensure that 8 bit and 16 bit versions of the same image can share the same prefix. We have at least one [TEM](https://en.wikipedia.org/wiki/Transmission_electron_microscopy) dataset, which is stored at [`jrc_dauer-larva.n5/em/tem-uint8`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_dauer-larva/jrc_dauer-larva.n5/em/tem-uint8/).
- Light microscopy images are stored under a prefix with the scheme `<dataset>.n5/lm/<identifier>/`, e.g. [`aic_desmosome-2.n5/lm/lm_488`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/aic_desmosome-2/aic_desmosome-2.n5/lm/lm_488/), [`jrc_cos7-11.n5/lm/er_palm`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_cos7-11/jrc_cos7-11.n5/lm/er_palm/). Note that the image identifiers are not consistently structured. Historically, light microscopy images have been rarer than electron microscopy images, and so we have been under less pressure to be consistent / strict about the naming of these images. This may change in the future, especially if we do an overhaul of our storage strategy.
- Segmentation, prediction, and other derived images are stored under a prefix with the scheme `<dataset>.n5/labels/<image-name>`, e.g. [`jrc_macrophage-2.n5/labels/mito_seg`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_macrophage-2/jrc_macrophage-2.n5/labels/mito_seg/), which is segmentation of mitochondria. Many of our machine learning models do not directly produce segmentations (i.e., images where sample values map to distinct semantic classes) -- instead, they emit a *prediction*, which is a scalar value indicating the model's estimate of the distance to the nearest instance of a target semantic class. Segmentation images are generated by thresholding these prediction images, and we currently store the predictions and segmentations together. So [`jrc_macrophage-2.n5/labels/mito_pred`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_macrophage-2/jrc_macrophage-2.n5/labels/mito_pred/) is the location of the prediction image that was used to generate [`jrc_macrophage-2.n5/labels/mito_seg`](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_macrophage-2/jrc_macrophage-2.n5/labels/mito_seg/).

#### Multiple image alignments

So far we have assumed (and ensured) that all the images in a dataset are aligned to the same coordinate space. This means that these images can all be viewed together in a single visualization tool. But image alignment is an art, and there may be multiple different, yet equally valid, alignments generated from the same raw images. With the layout described above, we cannot store different alignments of the same data under the constraint that all images in a dataset share the same coordinate space. This is a limitation that we plan to address when we overhaul the layout at some future date.

### Multiresolution image layout

Cellmap's imagery has features at many spatial scales, and we want users to be able to visualize these features interactively. A typical interactive visualization workflow might involve zooming out of an image to look for some large feature or region of interest, then zooming in to see fine-scale details. If we only published images at the highest level of detail, then the "zooming out" operation would force visualization programs to load an enormous amount of data, which would impair interactive visualization. 

We solve this problem in a very common way -- by generating an [image pyramid](https://en.wikipedia.org/wiki/Pyramid_%28image_processing%29), also known as a multiscale respresentation, from our images. This storage technique gives visualization clients access to smaller, coarser copies of the data, which they can load when a user is viewing a zoomed-out scene. We store the images as N5 datasets (arrays) inside an N5 group. You can see an example [here](https://open.quiltdata.com/b/janelia-cosem-datasets/tree/jrc_mus-liver/jrc_mus-liver.n5/labels/mito_seg/). Conventionally, the different scale level arrays are named `s0`, `s1`, `s2`..., where `s0` is the largest image (i.e., the original), `s1` was downscaled once, `s2` was downscaled twice, etc. 

Saving multiple arrays at different levels of detail addresses the interactive visualization problem, but it introduces some additional complications.

#### Multiscale image metadata

Instead of representing a single image as a single array in storage, we represent an image as a *collection* of arrays, that are similar in some ways (they have the same number of dimensions, and the same axis names) but different in others (they are different sizes, and their coordinates are different). To build software that works with multscale images, one must distinguish a multiscale image collection from a collection of random images that happen to be stored together, which typically entails declaring some special multiscale-specific metadata that consuming applications can correctly parse. 

Because Cellmap was using N5 for storing our published images and Neuroglancer for viewing them, we solved the "multiscale image metadata problem" by complying with the Neuroglancer N5 multiscale metadata convention, which you can find documented [here](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/n5). This repository contains Python code for creating / validating this metadata, which you can find documented [here](../api/multiscale/neuroglancer_n5.md).

Additionally, to ensure that Neuroglancer and other N5-fluent tools can open individual arrays with the correct coordinate scaling, each scale level array has a [`PixelResolution`](../api/multiscale/neuroglancer_n5.md#cellmap_schemas.multiscale.neuroglancer_n5.PixelResolution) defined in its attributes.

#### Downsampling and coordinates

The different scale levels of a multiscale image have different coordinates (locations in space), and the change in coordinates induced by downscaling depends on the method used to downscale. If we coarsen a 2D image by a factor of 2 along each axis by averaging contiguous, non-overlapping blocks of 2x2 samples, then it's obvious that adjacent samples of the resulting image are twice as far apart from each other than adjacent samples in the source image. But it's less obvious that the *origin* of the coarsened image has also moved relative to the origin of the source image, specifically, by 0.5 * the the inter-sample distance of the source image. In other words, this form of coarsening applies a *scaling*, and a *translation* to the coordinate grid of the coarsened image, relative to the coordinate grid of the source image. To further complicate matters, different downscaling techniques may apply different amounts of translation -- if, instead of taking a windowed average, we simply subsample the source image, then the translation applied by this operation depends on which samples we kept -- if we keep the "corner samples", including the origin, then there is no translation at all.

the Neuroglancer N5 multiscale conventions cannot express this information, because they do not support declaring translation transformations for each scale level. So we invented two pieces of metadata: The first, stored under the `transform` key in array attributes, specifies, for each axis, a name, unit, scaling, translation parameter. The purpose of this metadata is to compactly express a downscaled coordinate grid. This metadata is implemented in Python in this project; see the documentation [here](../api/multiscale/cosem.md#cellmap_schemas.multiscale.cosem.STTransform). The second piece of metadata is stored under the `multiscales` key in the attributes of the group containing the scale level arrays. `multiscales` is a list of JSON objects that provide, for each scale level array, a relative path and a transform object as described in the previous section. This metadata also has a Python implementation in this project; see its documentation [here](../api/multiscale/cosem.md#cellmap_schemas.cosem.multiscale.GroupMetadata)


When Cellmap started publishing our data there was no community standard for storing multiresolution images in a chunked format that satisfied our needs, so we pieced together our own conventions. With the advent of [OME-NGFF](https://ngff.openmicroscopy.org/), it's less and less attractive for us to maintain a home-grown convention when a community-maintained one exists. We intend to ultimately use OME-NGFF whereever we can, but this will require converting a *lot* of data, so it will take time.


