import os
import click


@click.command
def speedtest_cli():
    """
    Run a speedtest to check the internet connection.
    """
    exit_code = os.system(
        "curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -"
    )
    if exit_code != 0:
        print(
            f"Speedtest failed with exit code {exit_code}. Make sure you have curl and python installed."
        )
