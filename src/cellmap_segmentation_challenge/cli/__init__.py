import click

@click.group
def run():
    pass

@click.command
def echo():
    click.echo('hello')

run.add_command(echo, name='echo')