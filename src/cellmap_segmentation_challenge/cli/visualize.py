import click
import os

from upath import UPath
from cellmap_segmentation_challenge import SEARCH_PATH, RAW_NAME
from cellmap_segmentation_challenge.utils import load_safe_config
from cellmap_data.utils import find_level


@click.command
@click.option(
    "--datasets",
    "-d",
    type=click.STRING,
    required=True,
    default="*",
    help="Comma-separated list of datasets to view (Example: 'jrc_cos7-1a,jrc_cos7-2b') or '*' for all. Default: '*'.",
)
@click.option(
    "--crops",
    "-c",
    type=click.STRING,
    required=True,
    default="*",
    help="Comma-separated list of crops to view (Example: '111,112,113') or 'test' or 'all' for all. Default: '*'.",
)
@click.option(
    "--classes",
    "-C",
    type=click.STRING,
    required=True,
    default="*",
    help="Comma-separated list of label classes to view (Example: 'mito,er,nuc') or '*' for all. Default: '*'.",
)
@click.option(
    "--kinds",
    "-k",
    type=click.STRING,
    required=True,
    default="gt,predictions,processed,submission",
    help="Comma-separated list of kinds of data to view (Example: 'gt,processed'). Defaults to all: 'gt,predictions,processed,submission'. Raw (fibsem) data is always shown.",
)
def visualize_cli(datasets, crops, classes, kinds):
    """
    Visualize datasets and crops in Neuroglancer.

    Parameters
    ----------
    datasets : str
        Comma-separated list of datasets to view (Example: 'jrc_cos7-1a,jrc_cos7-2b') or '*' for all. Default: '*'.
    crops : str
        Comma-separated list of crops to view (Example: '111,112,113') or '*' for all. Default: '*'.
    classes : str
        Comma-separated list of label classes to view (Example: 'mito,er,nuc') or '*' for all. Default: '*'.
    kinds : str
        Comma-separated list of kinds of data to view (Example: 'gt,processed'). Defaults to all: 'gt,predictions,processed'. Raw (fibsem) data is always shown.
    """

    from cellmap_segmentation_challenge.visualize import visualize

    visualize(
        datasets=datasets.split(","),
        crops=crops.split(","),
        classes=classes.split(","),
        kinds=kinds.split(","),
    )


@click.command
@click.option(
    "--script_path",
    "-s",
    type=click.STRING,
    required=True,
    help="Path to the script to run for live prediction.",
)
@click.option(
    "--dataset",
    "-d",
    type=click.STRING,
    required=True,
    help="Dataset to view (Example: 'jrc_cos7-1a')",
)
@click.option(
    "--level",
    "-l",
    type=click.STRING,
    required=False,
    default=None,
    help="(Optional) Scale level to feed to model (Example: 's0'). If not specified, will be inferred from input_array_info in the config script.",
)
def flow(script_path, dataset, level):
    """
    Run a cellmap-flow to visualize live predictions using a script defining a model config, visualizing the results in Neuroglancer.

    Parameters
    ----------
    script_path : str
        Path to the script defining the model config (e.g. `examples/train_2D.py`).
    dataset : str
        Dataset to view (Example: 'jrc_cos7-1a'),
    level : str
        (Optional) Scale level to feed to model (Example: 's0'). If not specified, will be inferred from input_array_info in the config script.
    """

    config = load_safe_config(script_path)

    dataset_path = SEARCH_PATH.format(dataset=dataset, name=RAW_NAME)
    if level is None:
        level = find_level(
            dataset_path,
            {k: v for k, v in zip("zyx", config.input_array_info["scale"])},
        )

    os.system(
        f"cellmap_flow script -s {script_path} -d {(UPath(dataset_path) / level).path}"
    )
