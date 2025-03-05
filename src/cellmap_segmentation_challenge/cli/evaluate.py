import os
import click

from cellmap_segmentation_challenge.evaluate import (
    INSTANCE_CLASSES,
    PROCESSED_PATH,
    SUBMISSION_PATH,
    TRUTH_PATH,
)
from upath import UPath


@click.command
@click.option(
    "--submission_path",
    "-s",
    type=click.Path(exists=True),
    help="Path to the submission zip file",
    default=UPath(SUBMISSION_PATH).with_suffix(".zip").path,
)
@click.option(
    "--result_file",
    "-r",
    type=click.Path(),
    default="result.json",
    help="Path for the result json file. Defaults to 'result.json'",
)
@click.option(
    "--truth_path",
    "-t",
    type=click.Path(exists=True),
    default=TRUTH_PATH,
    help=f"Path to the ground truth data. Defaults to {TRUTH_PATH}",
)
@click.option(
    "--instance_classes",
    "-ic",
    type=click.STRING,
    default=None,
    help="A comma-separated list of class names that should be evaluated as instances. Defaults to None",
)
def evaluate_cli(submission_path, result_file, truth_path, instance_classes):
    """
    Evaluate a submission against the ground truth data.

    SUBMISSION_PATH: Path to the submission file
    """

    from cellmap_segmentation_challenge.evaluate import score_submission

    if instance_classes is not None:
        instance_classes = instance_classes.split(",")
    else:
        instance_classes = INSTANCE_CLASSES
    score_submission(submission_path, result_file, truth_path, instance_classes)


@click.command
@click.option(
    "--input_search_path",
    "-i",
    type=click.STRING,
    default=PROCESSED_PATH,
    help=f"Path to the prepared zarr data files. Defaults to {PROCESSED_PATH}",
    required=True,
)
@click.option(
    "--output_path",
    "-o",
    type=click.STRING,
    default=SUBMISSION_PATH,
    help=f"Directory to save the packaged submission. Defaults to {SUBMISSION_PATH}",
    required=True,
)
@click.option(
    "--overwrite",
    "-O",
    is_flag=True,
    help="Whether to overwrite the output submission file if it already exists",
)
@click.option(
    "--max_workers",
    "-w",
    type=click.INT,
    default=os.cpu_count(),
    help=f"The maximum number of workers to use for packaging the submission. Defaults to the number of CPUs on the system (currently {os.cpu_count()}).",
)
def package_submission_cli(input_search_path, output_path, overwrite, max_workers):
    """
    Package zarr datasets for submission.
    """

    from cellmap_segmentation_challenge.utils.submission import package_submission

    package_submission(input_search_path, output_path, overwrite, max_workers)
