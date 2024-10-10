import click
from ..evaluate import evaluate


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
    required=True,
    help="Path for the result json file",
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
