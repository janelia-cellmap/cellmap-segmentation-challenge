import click


@click.command
@click.argument(
    "config_path",
    type=click.Path(exists=True),
    required=True,
)
def train_cli(config_path):
    """
    Train a model using the configuration defined in the provided python file.

    CONFIG_PATH: Path to the python file defining the configuration to be used for training. The training will be executed in the same directory as the configuration file.
    """

    import os
    from upath import UPath

    from cellmap_segmentation_challenge.train import train

    os.chdir(UPath(config_path).parent.path)
    train(UPath(config_path).name)
