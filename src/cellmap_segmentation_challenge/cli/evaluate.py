import click
from ..evaluate import score_submission, TRUTH_PATH, INSTANCE_CLASSES


@click.command
@click.argument(
    "submission_path",
    type=click.Path(exists=True),
    required=True,
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
    if instance_classes is not None:
        instance_classes = instance_classes.split(",")
    else:
        instance_classes = INSTANCE_CLASSES
    score_submission(submission_path, result_file, truth_path, instance_classes)
