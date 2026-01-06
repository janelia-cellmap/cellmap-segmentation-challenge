import click

from .datasplit import get_dataset_counts_cli, make_datasplit_csv_cli
from .evaluate import evaluate_cli, package_submission_cli, batch_evaluate_cli
from .fetch_data import fetch_data_cli
from .predict import predict_cli
from .process import process_cli
from .train import train_cli
from .visualize import visualize_cli
from .speedtest import speedtest_cli


@click.group
def run():
    pass


run.add_command(fetch_data_cli, name="fetch-data")
run.add_command(train_cli, name="train")
run.add_command(predict_cli, name="predict")
run.add_command(process_cli, name="process")
run.add_command(evaluate_cli, name="evaluate")
run.add_command(make_datasplit_csv_cli, name="make-datasplit")
run.add_command(get_dataset_counts_cli, name="get-counts")
run.add_command(visualize_cli, name="visualize")
run.add_command(package_submission_cli, name="pack-results")
run.add_command(speedtest_cli, name="speedtest")
run.add_command(batch_evaluate_cli, name="batch-eval")
