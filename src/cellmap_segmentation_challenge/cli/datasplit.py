import click

from cellmap_segmentation_challenge.utils.datasplit import (
    CROP_NAME,
    RAW_NAME,
    SEARCH_PATH,
)


@click.command()
@click.option(
    "--classes",
    "-c",
    type=str,
    default="nuc,er",
    help="A comma-separated list of classes to include in the csv. Defaults to nuc,er.",
)
@click.option(
    "--scale",
    "-s",
    default=None,
    help="Single scalar or comma-separated list defining resolution (scale) used to filter out crops that don't have data at required scale. If only a scalar is specified, isotropic resolution is assumed. Default is not to filter data by resolution. Example: -s 1.0,2.0,3.0",
)
@click.option(
    "--force-all-classes",
    "-fa",
    is_flag=True,
    help="force all classes to be present in the training/validation datasets.",
)
@click.option(
    "--force-all-classes-train",
    "-ft",
    is_flag=True,
    help="Force all classes to be present in the training datasets.",
)
@click.option(
    "--force-all-classes-validate",
    "-fv",
    is_flag=True,
    help="Force all classes to be present in the validation datasets.",
)
@click.option(
    "--validate-ratio",
    "-vr",
    type=float,
    default=0.1,
    help="The ratio of the datasets to use for validation. Default is 0.1.",
)
@click.option(
    "--search_path",
    "-sp",
    type=str,
    default=SEARCH_PATH,
    help=f"The search path to use to find the datasets. Default is {SEARCH_PATH}.",
)
@click.option(
    "--raw_name",
    "-rn",
    type=str,
    default=RAW_NAME,
    help=f"The base name of the raw datasets. Default is {RAW_NAME}.",
)
@click.option(
    "--crop_name",
    "-cn",
    type=str,
    default=CROP_NAME,
    help=f"The base name of the crop datasets. Default is {CROP_NAME}.",
)
@click.option(
    "--csv_path",
    "-cp",
    type=str,
    default="datasplit.csv",
    help="The path to write the csv to. Default is datasplit.csv.",
)
@click.option(
    "--use_s3",
    "-s3",
    is_flag=True,
    help="Use s3 (remote) data stores instead of locally downloaded stores.",
)
@click.option(
    "--datasets",
    "-ds",
    type=str,
    default=None,
    help="A comma-separated list of dataset names to include in the csv. If not specified, all datasets will be included.",
)
@click.option(
    "--crops",
    "-cr",
    type=str,
    default=None,
    help="A comma-separated list of crop names to include in the csv. If not specified, all crops will be included. Example: -cr crop1,crop2,crop3",
)
def make_datasplit_csv_cli(
    classes,
    scale,
    force_all_classes,
    force_all_classes_train,
    force_all_classes_validate,
    validate_ratio,
    search_path,
    raw_name,
    crop_name,
    csv_path,
    use_s3,
    datasets,
    crops,
):
    """
    Make a datasplit csv file for the given classes with a given validate:train split ratio.
    """
    classes = classes.split(",")
    scale = None if scale is None else [float(s) for s in scale.split(",")]
    if scale is not None and len(scale) == 1:
        scale = [scale[0], scale[0], scale[0]]
    if force_all_classes_train:
        force_all_classes = "train"
        if force_all_classes_validate:
            force_all_classes = True
    elif force_all_classes_validate:
        force_all_classes = "validate"

    if datasets is not None:
        datasets = datasets.split(",")
    else:
        datasets = ["*"]

    if crops is not None:
        crops = crops.split(",")
    else:
        crops = ["*"]

    if use_s3:
        from cellmap_segmentation_challenge.utils.datasplit import make_s3_datasplit_csv

        make_s3_datasplit_csv(
            classes=classes,
            scale=scale,
            force_all_classes=force_all_classes,
            validation_prob=validate_ratio,
            csv_path=csv_path,
            datasets=datasets,
            crops=crops,
        )
    else:
        from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv

        make_datasplit_csv(
            classes=classes,
            scale=scale,
            force_all_classes=force_all_classes,
            validation_prob=validate_ratio,
            search_path=search_path,
            raw_name=raw_name,
            crop_name=crop_name,
            csv_path=csv_path,
            datasets=datasets,
            crops=crops,
        )


@click.command()
@click.option(
    "--classes",
    "-c",
    type=str,
    default="nuc,er",
    help="A comma-separated list of classes of which to count occurences. Defaults to nuc,er.",
)
@click.option(
    "--search_path",
    "-sp",
    type=str,
    default=SEARCH_PATH,
    help=f"The search path to use to find the datasets. Default is {SEARCH_PATH}.",
)
@click.option(
    "--raw_name",
    "-rn",
    type=str,
    default=RAW_NAME,
    help=f"The base name of the raw datasets. Default is {RAW_NAME}.",
)
@click.option(
    "--crop_name",
    "-cn",
    type=str,
    default=CROP_NAME,
    help=f"The base name of the crop datasets. Default is {CROP_NAME}.",
)
def get_dataset_counts_cli(classes, search_path, raw_name, crop_name):
    """Get the counts of each class in each dataset and print them to the stdout."""

    from cellmap_segmentation_challenge.utils.datasplit import get_dataset_counts

    classes = classes.split(",")
    dataset_class_counts = get_dataset_counts(
        classes=classes, search_path=search_path, raw_name=raw_name, crop_name=crop_name
    )

    # Print the counts
    print(f"Found {len(dataset_class_counts)} datasets.")
    for dataset_name, class_counts in dataset_class_counts.items():
        print(f"Dataset: {dataset_name}")
        for class_name, count in class_counts.items():
            print(f"\t{class_name}: \t{count}")
