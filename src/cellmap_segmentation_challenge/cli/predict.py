import click

from ..config import PREDICTIONS_PATH
from ..predict import predict


@click.command
@click.argument(
    "config_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--crops",
    "-c",
    type=click.STRING,
    required=True,
    default="test",
    help="Comma-separated list of crops to predict on (Example: '111,112,113') or 'test' or '*' for all. Default: 'test'.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.STRING,
    required=True,
    default=PREDICTIONS_PATH,
    help=f"Path to save the predicted crops with {'{crop}'} and {'{dataset}'} placeholders for formatting. Default: {PREDICTIONS_PATH}.",
)
@click.option(
    "--do-orthoplanes",
    "-do",
    type=click.BOOL,
    is_flag=True,
    required=False,
    default=True,
    help="Whether to predict the orthoplanes if the model is 2D. Default: True.",
)
@click.option(
    "--overwrite",
    "-O",
    type=click.BOOL,
    is_flag=True,
    required=False,
    default=False,
    help="Whether to overwrite the output path if it already exists. Default: False.",
)
def predict_cli(config_path, crops, output_path, do_orthoplanes, overwrite):
    """
    Predict the output of a model on a large dataset by splitting it into blocks and predicting each block separately.

    CONFIG_PATH: The path to the model configuration file. This can be the same as the config file used for training.
    """
    predict(
        config_path=config_path,
        crops=crops,
        output_path=output_path,
        do_orthoplanes=do_orthoplanes,
        overwrite=overwrite,
    )
