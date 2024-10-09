import click
from .fetch_data import fetch_data_cli
from .train import train_cli
from .predict import predict_cli, predict_ortho_planes_cli


@click.group
def run():
    pass


@click.command
def echo():
    click.echo("hello")


run.add_command(echo, name="echo")
run.add_command(fetch_data_cli, name="fetch-data")
run.add_command(train_cli, name="train")
run.add_command(predict_cli, name="predict")
run.add_command(predict_ortho_planes_cli, name="predict-ortho-planes")
