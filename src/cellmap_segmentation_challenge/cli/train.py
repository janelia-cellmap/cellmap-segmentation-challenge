import click
from ..train import train


@click.command
@click.argument(
    "config_path",
    type=click.Path(exists=True),
    required=True,
)
def train_cli(config_path):
    """
    Train a model using the configuration defined in the provided python file.

    CONFIG_PATH: Path to the python file defining the configuration to be used for training
    """
    train(config_path)
