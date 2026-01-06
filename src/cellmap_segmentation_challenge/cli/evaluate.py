import os
import click

from cellmap_segmentation_challenge.config import (
    PROCESSED_PATH,
    SUBMISSION_PATH,
    TRUTH_PATH,
    INSTANCE_CLASSES,
)
from upath import UPath

import logging

logger = logging.getLogger(__name__)


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
    help=f"A comma-separated list of class names that should be evaluated as instances. Defaults to {INSTANCE_CLASSES}",
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
    logger.info(f"Launching evaluation for submission: {submission_path}")
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

    logger.info(f"Packaging submission to: {output_path}")
    package_submission(input_search_path, output_path, overwrite, max_workers)


@click.command
@click.option(
    "--csv-path",
    "-c",
    type=click.Path(exists=True),
    help="Path to the CSV file containing submission paths and eval_IDs",
    required=True,
)
@click.option(
    "--cluster",
    "-C",
    is_flag=True,
    help="If set, submit evaluation jobs to a cluster instead of running locally. Will use default bsub command template.",
)
@click.option(
    "--num_cpus",
    "-n",
    type=click.INT,
    default=48,
    help="Number of CPUs to request per job when submitting to cluster. Default is 48.",
)
def batch_evaluate_cli(csv_path, cluster, num_cpus):
    from cellmap_segmentation_challenge.utils.batch_eval import eval_batch

    eval_batch(csv_path, cluster=cluster, num_cpus=num_cpus)
