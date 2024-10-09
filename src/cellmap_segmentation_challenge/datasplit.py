# %%
from glob import glob
import shutil
import sys
from typing import Optional
import numpy as np
import os

from tqdm import tqdm

SEARCH_PATH = (
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    + "/data/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/*/{label}"
)


def get_csv_string(
    path: str,
    classes: list[str],
    datapath_prefix: str,
    usage: str,
    raw_name: str,
    crops: Optional[list[str]] = None,
):
    dataset_name = path.removeprefix(datapath_prefix).split("/")[0]
    gt_name = "crop" + path.split("crop")[-1].split("/")[0]
    if crops is not None and gt_name not in crops:
        bar_string = gt_name + " not in crops, skipping"
        return None, bar_string
    gt_path = path.removesuffix(gt_name).rstrip("/")
    raw_path = os.path.join(datapath_prefix, dataset_name, f"{dataset_name}.zarr")
    if not os.path.exists(os.path.join(raw_path, raw_name)):
        bar_string = f"No raw data found for {dataset_name} at {os.path.join(raw_path, raw_name)}, trying n5 format"
        raw_path = os.path.join(datapath_prefix, dataset_name, f"{dataset_name}.n5")
        if not os.path.exists(os.path.join(raw_path, raw_name)):
            bar_string = f"No raw data found for {dataset_name} at {os.path.join(raw_path, raw_name)}, skipping"
            return None, bar_string
    bar_string = (
        f"Found raw data for {dataset_name} at {os.path.join(raw_path, raw_name)}"
    )
    return (
        f'"{usage}","{raw_path}","{raw_name}","{gt_path}","{gt_name+os.path.sep}[{",".join([c for c in classes])}]"\n',
        bar_string,
    )


def make_datasplit_csv(
    classes: list[str] = ["nuc", "mito"],
    force_all_classes: bool | str = False,
    validation_prob: float = 0.3,
    datasets: list[str] = ["*"],
    search_path: str = SEARCH_PATH,
    raw_name: str = "recon-1/em/fibsem-uint8",
    csv_path: str = "datasplit.csv",
    dry_run: bool = False,
    crops: Optional[list[str]] = None,
):
    print(f"Classes: {classes}")
    print(f"Force all classes: {force_all_classes}")
    print(f"Validation probability: {validation_prob}")
    print(f"Datasets: {datasets}")
    print(f"Search path: {search_path}")
    print(f"Raw name: {raw_name}")
    print(f"CSV path: {csv_path}")
    print(f"Dry run: {dry_run}")
    print(f"Crops: {crops}")

    # Define the paths to the raw and groundtruth data and the label classes by crawling the directories and writing the paths to a csv file
    datapath_prefix = search_path.split("{")[0]
    datapaths = {}
    for dataset in datasets:
        for label in classes:
            these_datapaths = glob(search_path.format(dataset=dataset, label=label))
            these_datapaths = [
                path.removesuffix(os.path.sep + label) for path in these_datapaths
            ]
            for path in these_datapaths:
                if path not in datapaths:
                    datapaths[path] = []
                datapaths[path].append(label)

    if dry_run:
        print("Dry run, not writing csv")
        return datapaths

    shutil.rmtree(csv_path, ignore_errors=True)
    assert not os.path.exists(csv_path), f"CSV file {csv_path} already exists"

    usage_dict = {
        k: "train" if np.random.rand() > validation_prob else "validate"
        for k in datapaths.keys()
    }
    num_train = num_validate = 0
    bar = tqdm(datapaths.keys())
    for path in bar:
        print(f"Processing {path}")
        usage = usage_dict[path]
        if force_all_classes == usage:
            if len(datapaths[path]) != len(classes):
                usage = "train" if usage == "validate" else "validate"
        elif force_all_classes is True:
            if len(datapaths[path]) != len(classes):
                usage_dict[path] = "none"
                continue
        usage_dict[path] = usage

        csv_string, bar_string = get_csv_string(
            path, datapaths[path], datapath_prefix, usage, raw_name, crops
        )
        bar.set_postfix_str(bar_string)
        if csv_string is not None:
            with open(csv_path, "a") as f:
                if csv_string is not None:
                    f.write(csv_string)
            if usage == "train":
                num_train += 1
            else:
                num_validate += 1

    assert num_train + num_validate > 0, "No datasets found"
    print(f"Number of datasets: {num_train + num_validate}")
    print(
        f"Number of training datasets: {num_train} ({num_train/(num_train+num_validate)*100:.2f}%)"
    )
    print(
        f"Number of validation datasets: {num_validate} ({num_validate/(num_train+num_validate)*100:.2f}%)"
    )
    print(f"CSV written to {csv_path}")


def get_dataset_counts(
    classes: list[str] = ["nuc", "mito"],
    search_path: str = SEARCH_PATH,
    raw_name: str = "recon-1/em/fibsem-uint8",
):
    # Count the # of crops per class per dataset
    datapath_prefix = search_path.split("*")[0]
    dataset_class_counts = {}
    for label in classes:
        these_datapaths = glob(search_path.format(label=label))
        these_datapaths = [
            path.removesuffix(os.path.sep + label) for path in these_datapaths
        ]
        for path in these_datapaths:
            dataset_name = path.removeprefix(datapath_prefix).split("/")[0]
            raw_path = os.path.join(
                datapath_prefix, dataset_name, f"{dataset_name}.zarr"
            )
            if not os.path.exists(os.path.join(raw_path, raw_name)):
                print(
                    f"No raw data found for {dataset_name} at {os.path.join(raw_path, raw_name)}, trying n5 format"
                )
                continue
            if dataset_name not in dataset_class_counts:
                dataset_class_counts[dataset_name] = {}
            if label not in dataset_class_counts[dataset_name]:
                dataset_class_counts[dataset_name][label] = 1
            else:
                dataset_class_counts[dataset_name][label] += 1

    return dataset_class_counts


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1][0] == "[":
        classes = sys.argv[1][1:-1].split(",")
    elif len(sys.argv) > 1:
        search_path = sys.argv[1]
        classes = ["nuc", "er"]
    else:
        classes = ["nuc", "er"]
        search_path = "./data.zarr/{dataset}/{label}"

    os.remove("datasplit.csv")

    make_datasplit_csv(classes=classes, search_path=search_path, validation_prob=0.15)
