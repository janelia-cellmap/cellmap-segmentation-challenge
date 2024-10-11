import click

from cellmap_segmentation_challenge.utils.datasplit import (
    make_datasplit_csv,
    get_dataset_counts,
    SEARCH_PATH,
    RAW_NAME,
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
    help="The search path to use to find the datasets. Default is {SEARCH_PATH}.",
)
@click.option(
    "--raw_name",
    "-rn",
    type=str,
    default=RAW_NAME,
    help=f"The name of the raw dataset. Default is {RAW_NAME}.",
)
@click.option(
    "--csv_path",
    "-cp",
    type=str,
    default="datasplit.csv",
    help="The path to write the csv to. Default is datasplit.csv.",
)
def make_datasplit_csv_cli(
    classes,
    force_all_classes,
    force_all_classes_train,
    force_all_classes_validate,
    validate_ratio,
    search_path,
    raw_name,
    csv_path,
):
    """
    Make a datasplit csv file for the given classes and datasets.
    """
    classes = classes.split(",")
    if force_all_classes_train:
        force_all_classes = "train"
        if force_all_classes_validate:
            force_all_classes = True
    elif force_all_classes_validate:
        force_all_classes = "validate"

    make_datasplit_csv(
        classes=classes,
        force_all_classes=force_all_classes,
        validation_prob=validate_ratio,
        search_path=search_path,
        raw_name=raw_name,
        csv_path=csv_path,
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
    help=f"The name of the raw dataset. Default is {RAW_NAME}.",
)
def get_dataset_counts_cli(classes, search_path, raw_name):
    """Get the counts of each class in each dataset and print them to the stdout."""
    classes = classes.split(",")
    dataset_class_counts = get_dataset_counts(
        classes=classes, search_path=search_path, raw_name=RAW_NAME, raw_name=raw_name
    )

    # Print the counts
    print(f"Found {len(dataset_class_counts)} datasets.")
    for dataset_name, class_counts in dataset_class_counts.items():
        print(f"Dataset: {dataset_name}")
        for class_name, count in class_counts.items():
            print(f"\t{class_name}: \t{count}")
