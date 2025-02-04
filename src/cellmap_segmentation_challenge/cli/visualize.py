import click
from cellmap_segmentation_challenge.visualize import SEARCH_PATHS

kinds = ",".join(SEARCH_PATHS.keys())


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
    default=kinds,
    help=f"Comma-separated list of kinds of data to view (Example: 'gt,processed'). Defaults to all: '{kinds}'. Raw (fibsem) data is always shown.",
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
