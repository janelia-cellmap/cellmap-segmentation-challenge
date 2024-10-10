import click
from ..evaluate import evaluate


@click.command
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to the python file defining the configuration to be used for training",
)
@click.option(
    "--input_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input data",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to the output data",
)
@click.option(
    "--input_block_shape",
    "-s",
    type=click.STRING,
    required=True,
    help="A comma-separated list of describing the shape of the data blocks input to the model",
)
@click.option(
    "--channels",
    "-c",
    type=click.STRING,
    required=True,
    help="A comma-separated list of channel names in order according to the output of the model",
)
def evaluate_cli(model, input_path, output_path, input_block_shape, channels):
    input_block_shape = input_block_shape.split(",")
    channels = channels.split(",")
    evaluate(...
