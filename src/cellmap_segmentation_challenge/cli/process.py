import os
import click

from cellmap_segmentation_challenge.config import PREDICTIONS_PATH, PROCESSED_PATH


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
    default=PREDICTIONS_PATH,
    help=f"Path to save the processed crops with {'{crop}'} and {'{dataset}'} placeholders for formatting. Default: {PREDICTIONS_PATH}.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.STRING,
    required=True,
    default=PROCESSED_PATH,
    help=f"Path to save the processed crops with {'{crop}'} and {'{dataset}'} placeholders for formatting. Default: {PROCESSED_PATH}.",
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
@click.option(
    "--device",
    "-d",
    type=click.STRING,
    required=False,
    default=None,
    help="Device to use for processing the data. Default: None.",
)
@click.option(
    "--max-workers",
    "-w",
    type=click.INT,
    required=False,
    default=os.cpu_count(),
    help=f"Maximum number of workers to use for processing the data. Defaults to the number of CPUs on the system (currently {os.cpu_count()}).",
)
def process_cli(
    config_path, crops, input_path, output_path, overwrite, device, max_workers
):
    """
    Process data from a large dataset by splitting it into blocks and processing each block separately.

    CONFIG_PATH: The path to the processing configuration file.
    """

    from cellmap_segmentation_challenge.process import process

    process(
        config_path=config_path,
        crops=crops,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        device=device,
        max_workers=max_workers,
    )
