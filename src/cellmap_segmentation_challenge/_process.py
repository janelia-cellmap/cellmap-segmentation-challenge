from upath import UPath


def process(config_path: str | UPath) -> None:
    """
    Process and save arrays using an arbitrary process function defined in a config python file.

    Parameters
    ----------
    config_path : str | UPath
        The path to the python file containing the process function and other configurations. The script should specify the process function as `process_func` and any other required configurations.
    """
