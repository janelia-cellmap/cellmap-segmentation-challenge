import click
from upath import UPath

from ..config import REPO_ROOT
from ..process import process


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
    "--input-path",
    "-i",
    type=click.STRING,
    required=True,
    default=(REPO_ROOT / "data/predictions/{dataset}.zarr/{crop}").path,
    help=f"Path to save the processed crops with {'{crop}'} and {'{dataset}'} placeholders for formatting. Default: {(REPO_ROOT / 'data/predictions/{dataset}.zarr/{crop}').path}.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.STRING,
    required=True,
    default=(REPO_ROOT / "data/processed/{dataset}.zarr/{crop}").path,
    help=f"Path to save the processed crops with {'{crop}'} and {'{dataset}'} placeholders for formatting. Default: {(REPO_ROOT / 'data/processed/{dataset}.zarr/{crop}').path}.",
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
def process_cli(config_path, crops, input_path, output_path, overwrite):
    """
    Process data from a large dataset by splitting it into blocks and processing each block separately.

    CONFIG_PATH: The path to the processing configuration file.
    """
    process(
        config_path=config_path,
        crops=crops,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
    )
