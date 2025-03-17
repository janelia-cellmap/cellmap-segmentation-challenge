# %%
import csv
import os
import shutil
import sys
from glob import glob
from typing import Optional

import numpy as np
from tqdm import tqdm
from upath import UPath
import zarr

from ..config import (
    CROP_NAME,
    RAW_NAME,
    SEARCH_PATH,
    S3_CROP_NAME,
    S3_RAW_NAME,
    S3_SEARCH_PATH,
    GT_S3_BUCKET,
    RAW_S3_BUCKET,
)


# TODO: Consolidate with get_formatted_fields
def get_dataset_name(
    raw_path: str, search_path: str = SEARCH_PATH, raw_name: str = RAW_NAME
) -> str:
    """
    Get the name of the dataset from the raw path.
    """
    path_base = search_path.format(dataset="{dataset}", name=raw_name)
    assert "{dataset}" in path_base, (
        f"search_path {search_path} must contain" + "{dataset}"
    )
    for rp, sp in zip(
        UPath(raw_path).path.split("/"), UPath(path_base).path.split("/")
    ):
        if sp == "{dataset}":
            return rp
    raise ValueError(
        f"Could not find dataset name in {raw_path} with {search_path} as template"
    )


# TODO: Consolidate with get_formatted_fields
def get_raw_path(crop_path: str, raw_name: str = RAW_NAME, label: str = "") -> str:
    """
    Get the path to the raw data for a given crop path.

    Parameters
    ----------
    crop_path : str
        The path to the crop.
    raw_name : str, optional
        The name of the raw data, by default RAW_NAME
    label : str, optional
        The label class at the crop_path, by default ""

    Returns
    -------
    str
        The path to the raw data.
    """
    crop_path = crop_path.rstrip(label + os.path.sep)
    crop_name = CROP_NAME.format(crop=os.path.basename(crop_path), label="").rstrip(
        os.path.sep
    )
    return (UPath(crop_path.removesuffix(crop_name)) / raw_name).path


def get_formatted_fields(
    path: str, base_path: str, fields: list[str]
) -> dict[str, str]:
    """
    Get the formatted fields from the path.

    Parameters
    ----------
    path : str
        The path to get the fields from.
    base_path : str
        The unformatted path to find the fields in.
    fields : list[str]
        The fields to get from the path.

    Returns
    -------
    dict[str, str]
        The formatted fields.
    """
    field_results = {}
    for rp, sp in zip(UPath(path).path.split("/"), UPath(base_path).path.split("/")):
        for field in fields:
            if (
                field in sp and field.strip("{}") not in field_results
            ):  # Will only keep first result
                remainders = sp.split(field)
                result = rp.removeprefix(remainders[0]).removesuffix(remainders[1])
                field_results[field.strip("{}")] = result
    return field_results


def get_s3_csv_string(path: str, classes: list[str], usage: str):
    """
    Get the csv string for a given dataset path, to be written to the datasplit csv file.

    Parameters
    ----------
    path : str
        The path to the dataset.
    classes : list[str]
        The classes present in the dataset.
    usage : str
        The usage of the dataset (train or validate).

    Returns
    -------
    str
        The csv string for the dataset.
    """
    dataset_name = get_formatted_fields(path, S3_SEARCH_PATH, ["dataset", "name"])[
        "dataset"
    ]
    raw_path = UPath("s3://" + RAW_S3_BUCKET, anon=True) / S3_SEARCH_PATH.format(
        dataset=dataset_name, name=S3_RAW_NAME
    )

    raw_zarr_path = raw_path.path.split(".zarr")[0] + ".zarr"
    gt_zarr_path = (UPath("s3://" + GT_S3_BUCKET, anon=True) / path).path.split(
        ".zarr"
    )[0] + ".zarr"
    raw_ds_name = raw_path.path.removeprefix(raw_zarr_path + os.path.sep)
    gt_ds_name = path.split(".zarr")[-1].removeprefix(os.path.sep)
    bar_string = f"Found raw data for {dataset_name} at {raw_path}"
    return (
        f'"{usage}","{"s3://" + raw_zarr_path}","{raw_ds_name}","{"s3://" + gt_zarr_path}","{gt_ds_name+os.path.sep}[{",".join([c for c in classes])}]"\n',
        bar_string,
    )


def get_csv_string(
    path: str,
    classes: list[str],
    usage: str,
    raw_name: str = RAW_NAME,
    search_path: str = SEARCH_PATH,
):
    """
    Get the csv string for a given dataset path, to be written to the datasplit csv file.

    Parameters
    ----------
    path : str
        The path to the dataset.
    classes : list[str]
        The classes present in the dataset.
    usage : str
        The usage of the dataset (train or validate).
    raw_name : str, optional
        The name of the raw data. Default is RAW_NAME.
    search_path : str, optional
        The search path to use to find the datasets. Default is SEARCH_PATH.

    Returns
    -------
    str
        The csv string for the dataset.
    """
    raw_path = get_raw_path(path, raw_name)
    dataset_name = get_dataset_name(
        raw_path, search_path=search_path, raw_name=raw_name
    )

    if not UPath(raw_path).exists():
        bar_string = (
            f"No raw data found for {dataset_name} at {raw_path}, trying n5 format"
        )
        raw_path = raw_path.replace(".zarr", ".n5")
        if not UPath(raw_path).exists():
            bar_string = f"No raw data found for {dataset_name} at {raw_path}, skipping"
            return None, bar_string
        zarr_path = raw_path.split(".n5")[0] + ".n5"
    else:
        zarr_path = raw_path.split(".zarr")[0] + ".zarr"
    raw_ds_name = raw_path.removeprefix(zarr_path + os.path.sep)
    gt_ds_name = path.removeprefix(zarr_path + os.path.sep)
    bar_string = f"Found raw data for {dataset_name} at {raw_path}"
    return (
        f'"{usage}","{zarr_path}","{raw_ds_name}","{zarr_path}","{gt_ds_name+os.path.sep}[{",".join([c for c in classes])}]"\n',
        bar_string,
    )


def make_s3_datasplit_csv(
    classes: list[str] = ["nuc", "mito"],
    scale : list[float] = None,
    force_all_classes: bool | str = False,
    validation_prob: float = 0.1,
    datasets: list[str] = ["*"],
    crops: list[str] = ["*"],
    csv_path: str = "datasplit.csv",
    dry_run: bool = False,
    **kwargs,
):
    """
    Make a datasplit csv file for the given classes and datasets.

    Parameters
    ----------
    classes : list[str], optional
        The classes to include in the csv, by default ["nuc", "mito"]
    force_all_classes : bool | str, optional
        If True, force all classes to be present in the training/validation datasets. If False, as long as at least one requested class is present, a crop will be included. If "train" or "validate", force all classes to be present in the training or validation datasets, respectively. By default False.
    validation_prob : float, optional
        The probability of a dataset being in the validation set, by default 0.1
    datasets : list[str], optional
        The datasets to include in the csv, by default ["*"], which includes all datasets
    crops : list[str], optional
        The crops to include in the csv, by default all crops are included. Otherwise, only the crops in the list are included.
    csv_path : str, optional
        The path to write the csv file to, by default "datasplit.csv"
    dry_run : bool, optional
        If True, do not write the csv file - just return the found datapaths. By default False
    **kwargs : dict
        Additional keyword arguments will be unused. Kept for compatibility with make_datasplit_csv.
    """
    # Define the paths to the raw and groundtruth data and the label classes by crawling the directories and writing the paths to a csv file
    if not dry_run:
        shutil.rmtree(csv_path, ignore_errors=True)
        assert not os.path.exists(
            csv_path
        ), f"CSV file {csv_path} already exists and could not be overwritten"
    datapaths = {}
    for dataset in datasets:
        for crop in crops:
            for label in classes:
                these_datapaths = list(
                    UPath("s3://" + GT_S3_BUCKET, anon=True).glob(
                        S3_SEARCH_PATH.format(
                            dataset=dataset,
                            name=S3_CROP_NAME.format(crop=crop, label=label),
                        )
                    )
                )
                if len(these_datapaths) == 0:
                    continue
                these_datapaths = [
                    path.path.removesuffix(os.path.sep + label).removeprefix(
                        GT_S3_BUCKET + os.path.sep
                    )
                    for path in these_datapaths
                ]
                for path in these_datapaths:
                    if path not in datapaths:
                        datapaths[path] = []
                        
                    #check scale
                    s3_path = UPath("s3://" + GT_S3_BUCKET + '/' + path + '/' + label, anon=True)
                    store = zarr.storage.FSStore(s3_path)
                    zg = zarr.open(store, mode="r")
                    scale_label = zg.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
                    if all([True if sc_label <= sc else False for sc_label, sc in zip(scale_label, scale) ]):
                        datapaths[path].append(label)

    if dry_run:
        print("Dry run, not writing csv")
        return datapaths

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

        csv_string, bar_string = get_s3_csv_string(path, datapaths[path], usage)
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


def make_datasplit_csv(
    classes: list[str] = ["nuc", "mito"],
    scale: list[float] = [pow(10, 3),]*3,
    force_all_classes: bool | str = False,
    validation_prob: float = 0.1,
    datasets: list[str] = ["*"],
    crops: list[str] = ["*"],
    search_path: str = SEARCH_PATH,
    raw_name: str = RAW_NAME,
    crop_name: str = CROP_NAME,
    csv_path: str = "datasplit.csv",
    dry_run: bool = False,
):
    """
    Make a datasplit csv file for the given classes and datasets.

    Parameters
    ----------
    classes : list[str], optional
        The classes to include in the csv, by default ["nuc", "mito"]
    scale : list[float]
        Filter crops by annotation scale, i.e. only fibsem data that has groundtruth array with this scale will be included in the datasplit.
    force_all_classes : bool | str, optional
        If True, force all classes to be present in the training/validation datasets. If False, as long as at least one requested class is present, a crop will be included. If "train" or "validate", force all classes to be present in the training or validation datasets, respectively. By default False.
    validation_prob : float, optional
        The probability of a dataset being in the validation set, by default 0.1
    datasets : list[str], optional
        The datasets to include in the csv, by default ["*"], which includes all datasets
    crops : list[str], optional
        The crops to include in the csv, by default all crops are included. Otherwise, only the crops in the list are included.
    search_path : str, optional
        The search path to use to find the datasets, by default SEARCH_PATH
    raw_name : str, optional
        The name of the raw data, by default RAW_NAME
    crop_name : str, optional
        The name of the crop, by default CROP_NAME
    csv_path : str, optional
        The path to write the csv file to, by default "datasplit.csv"
    dry_run : bool, optional
        If True, do not write the csv file - just return the found datapaths. By default False
    """
    # Define the paths to the raw and groundtruth data and the label classes by crawling the directories and writing the paths to a csv file
    datapaths = {}
    for dataset in datasets:
        for crop in crops:
            for label in classes:
                these_datapaths = glob(
                    search_path.format(
                        dataset=dataset, name=crop_name.format(crop=crop, label=label)
                    )
                )
                if len(these_datapaths) == 0:
                    continue
                these_datapaths = [
                    path.removesuffix(os.path.sep + label) for path in these_datapaths
                ]
                for path in these_datapaths:
                    if path not in datapaths:
                        datapaths[path] = []
                    
                    #check scale
                    zg = zarr.open(os.path.join(path, label))
                    scale_label = zg.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
                    if all([True if sc_label <= sc else False for sc_label, sc in zip(scale_label, scale) ]):
                        datapaths[path].append(label)

    if dry_run:
        print("Dry run, not writing csv")
        return datapaths

    shutil.rmtree(csv_path, ignore_errors=True)
    assert not os.path.exists(
        csv_path
    ), f"CSV file {csv_path} already exists and cannot be overwritten"

    usage_dict = {
        k: "train" if np.random.rand() > validation_prob else "validate"
        for k in datapaths.keys()
    }
    # Now enforce that there is one training and one validation crop if possible
    if len(usage_dict) >= 2:
        if np.sum(usage_dict.values() == "train") == 0:
            usage_dict[list(usage_dict.keys())[0]] = "train"
        elif np.sum(usage_dict.values() == "validate") == 0:
            usage_dict[list(usage_dict.keys())[0]] = "validate"
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
            path, datapaths[path], usage, raw_name, search_path
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
    raw_name: str = RAW_NAME,
    crop_name: str = CROP_NAME,
):
    """
    Get the counts of each class in each dataset.

    Parameters
    ----------
    classes : list[str], optional
        The classes to include in the csv, by default ["nuc", "mito"]
    search_path : str, optional
        The search path to use to find the datasets, by default SEARCH_PATH
    raw_name : str, optional
        The name of the raw data, by default RAW_NAME
    crop_name : str, optional
        The name of the crop, by default CROP_NAME

    Returns
    -------
    dict
        A dictionary of the counts of each class in each dataset.
    """
    dataset_class_counts = {}
    for label in classes:
        these_datapaths = glob(
            search_path.format(
                dataset="*", name=crop_name.format(crop="*", label=label)
            )
        )
        for path in these_datapaths:
            raw_path = get_raw_path(path, raw_name, label)
            dataset_name = get_dataset_name(raw_path)
            if not UPath(raw_path).exists():
                print(
                    f"No raw data found for {dataset_name} at {raw_path}, trying n5 format"
                )
                raw_path = raw_path.replace(".zarr", ".n5")
                if not UPath(raw_path).exists():
                    print(
                        f"No raw data found for {dataset_name} at {raw_path}, skipping"
                    )
                    continue
            if dataset_name not in dataset_class_counts:
                dataset_class_counts[dataset_name] = {}
            if label not in dataset_class_counts[dataset_name]:
                dataset_class_counts[dataset_name][label] = 1
            else:
                dataset_class_counts[dataset_name][label] += 1

    return dataset_class_counts


def get_tested_classes(
    csv_path: str = (UPath(__file__).parent / "tested_classes.csv").path,
):
    """
    Get the classes that will be tested for the challenge.

    Parameters
    ----------
    csv_path : str, optional
        The path to the csv file, by default "tested_classes.csv"

    Returns
    -------
    list[str]
        A list of the classes that have been tested.
    """
    tested_classes = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            tested_classes.append(row[0])
    return tested_classes


def get_class_incl_ids(incl_ids_string):
    if incl_ids_string is None or incl_ids_string == "":
        return []
    return [int(id) for id in incl_ids_string.split(",")]


def get_class_relations(
    csv_path: str = (UPath(__file__).parent / "classes.csv").path,
    named_classes: Optional[list[str]] = None,
):
    classes_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if named_classes is not None and row[0] not in named_classes:
                continue
            id = int(row[1])
            if id not in classes_dict:
                classes_dict[id] = {}
            classes_dict[id]["name"] = row[0]
            classes_dict[id]["incl_ids"] = set(get_class_incl_ids(*row[2:]) + [id])

    # Get all potentially overlapping classes (e.g. cell can overlap with every organelle but not ecs)
    class_relation_dict = {}
    for id1, info1 in classes_dict.items():
        if info1["name"] not in class_relation_dict:
            class_relation_dict[info1["name"]] = set()
        for id2, info2 in classes_dict.items():
            if id1 == id2:
                class_relation_dict[info1["name"]].add(info2["name"])
            if len(info1["incl_ids"].intersection(info2["incl_ids"])) > 0:
                class_relation_dict[info1["name"]].add(info2["name"])

    class_ids = set(v["name"] for v in classes_dict.values())
    for key, value in class_relation_dict.items():
        class_relation_dict[key] = class_ids - value

    return class_relation_dict


if __name__ == "__main__":
    """
    Usage: python datasplit.py [search_path] [classes]

    search_path: The search path to use to find the datasets. Defaults to SEARCH_PATH.
    classes: A comma-separated list of classes to include in the csv. Defaults to ["nuc", "er"].
    """
    if len(sys.argv) > 1 and sys.argv[1][0] == "[":
        classes = sys.argv[1][1:-1].split(",")
        if len(sys.argv) > 2:
            search_path = sys.argv[2]
        else:
            search_path = SEARCH_PATH
    elif len(sys.argv) > 1:
        search_path = sys.argv[1]
        classes = ["nuc", "er"]
    else:
        classes = ["nuc", "er"]
        search_path = SEARCH_PATH

    os.remove("datasplit.csv")

    make_datasplit_csv(classes=classes, search_path=search_path, validation_prob=0.1)
