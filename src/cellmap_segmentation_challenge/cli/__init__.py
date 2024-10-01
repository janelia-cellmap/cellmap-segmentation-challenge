import click
from .fetch_data import fetch_data_cli

@click.group
def run():
    pass

@click.command
def echo():
    click.echo('hello')

run.add_command(echo, name='echo')
run.add_command(fetch_data_cli, name='fetch-data')