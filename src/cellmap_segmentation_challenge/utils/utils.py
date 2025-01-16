from cellmap_segmentation_challenge.utils import get_tested_classes
import zarr

from upath import UPath


def format_coordinates(coordinates):
    """
    Format the coordinates to a string.

    Parameters
    ----------
    coordinates : list
        List of coordinates.

    Returns
    -------
    str
        Formatted string.
    """
    return f"[{';'.join([str(c) for c in coordinates])}]"


def construct_test_crop_manifest(
    path_root: str,
    search_path: str = "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}",
    write_path: str = (UPath(__file__).parent / "test_crop_manifest.csv").path,
) -> None:
    """
    Construct a manifest file for testing crops from a given path.

    Parameters
    ----------
    path_root : str
        Path to the directory containing the datasets. File tree should be as follows:
        path
        ├── dataset_1
        │   ├── crop_1
        │   │   ├── label_1
        │   │   │   ├── .zattrs

    search_path : str, optional
        Format string to search for the crops. The default is "{path_root}/{dataset}/groundtruth.zarr/{crop}/{label}". The function assumes that the keys appear in the file tree in the following order: 1) "path_root", 2) "dataset", 3) "crop", 4) "label"
    """
    # Get the tested classes
    tested_classes = set(get_tested_classes())

    # Construct the manifest
    manifest = [
        "crop_name,dataset,class_label,voxel_size,translation,shape",
    ]

    # Get datasets
    datasets = [
        d.name
        for d in UPath(
            search_path.split("{dataset}")[0].format(path_root=path_root)
        ).iterdir()
        if d.is_dir()
    ]

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Get crops
        crops = [
            d.name
            for d in UPath(
                search_path.split("{crop}")[0].format(
                    path_root=path_root, dataset=dataset
                )
            ).iterdir()
            if d.is_dir()
        ]
        for crop in crops:
            print(f"\tProcessing crop: {crop}")
            # Get labels in crop
            had_classes = set(
                [
                    d.name
                    for d in UPath(
                        search_path.split("{label}")[0].format(
                            path_root=path_root, dataset=dataset, crop=crop
                        )
                    ).iterdir()
                    if d.is_dir()
                ]
            )

            # Filter for tested classes
            labels = list(had_classes.intersection(tested_classes))

            for label in labels:
                print(f"\t\tProcessing label: {label}")
                # Get the zarr file
                zarr_file = zarr.open(
                    search_path.format(
                        path_root=path_root, dataset=dataset, crop=crop, label=label
                    ),
                    mode="r",
                )

                # Get the metadata
                metadata = zarr_file.attrs.asdict()["multiscales"][0]["datasets"][0][
                    "coordinateTransformations"
                ]
                for meta in metadata:
                    if meta["type"] == "translation":
                        translation = format_coordinates(meta["translation"])
                    elif meta["type"] == "scale":
                        voxel_size = format_coordinates(meta["scale"])
                shape = format_coordinates(zarr_file["s0"].shape)
                manifest.append(
                    f"{crop.replace('crop', '')},{dataset},{label},{voxel_size},{translation},{shape}"
                )
                print(
                    f"\t\t\tScale: {voxel_size}\n\t\t\tTranslation: {translation}\n\t\t\tShape: {shape}"
                )

    # Write the manifest
    with open(write_path, "w") as f:
        f.write("\n".join(manifest))

    print(f"Manifest written to: {write_path}")
