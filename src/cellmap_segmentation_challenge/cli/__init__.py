import click
from .fetch_data import fetch_crops_cli

@click.group
def run():
    pass

@click.command
def echo():
    click.echo('hello')

run.add_command(echo, name='echo')
run.add_command(fetch_crops_cli, name='fetch-crops')