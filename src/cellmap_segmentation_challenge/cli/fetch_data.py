import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy as np
import structlog
import zarr
from dotenv import load_dotenv
from pydantic_zarr.v2 import GroupSpec
from upath import UPath as Path
from xarray import DataArray  # TODO: Add lazy import
from xarray_ome_ngff import read_multiscale_group
from xarray_ome_ngff.v04.multiscale import transforms_from_coords, VectorScale
from yarl import URL
from zarr.storage import FSStore

from cellmap_segmentation_challenge.utils.crops import (
    CropRow,
    TestCropRow,
    fetch_manifest,
    ZipDatasetRow,
    fetch_zip_manifest,
    get_test_crops,
)

from cellmap_segmentation_challenge.utils.fetch_data import (
    _resolve_em_dest_path,
    _resolve_gt_dest_path,
    get_chunk_keys,
    partition_copy_store,
    read_group,
    subset_to_slice,
    get_zip_if_available,
    download_file_with_progress,
)
from cellmap_segmentation_challenge.config import BASE_DATA_PATH

load_dotenv()


@click.command
@click.option(
    "-c",
    "--crops",
    type=click.STRING,
    required=True,
    default="all",
    help='A comma-separated list of crops to download, e.g., "111,112,113", "test" to only download test crops, or "all" to download all crops. Default: "all".',
)
@click.option(
    "-p",
    "--raw-padding",
    type=click.INT,
    default=0,
    help="Padding to apply to raw data, in voxels. Default: 0.",
)
@click.option(
    "-d",
    "--dest",
    type=click.STRING,
    default=BASE_DATA_PATH.path,
    help=f"Path to directory where data will be stored. Defaults to {BASE_DATA_PATH.path}",
)
@click.option(
    "-m",
    "--access-mode",
    type=click.STRING,
    default="append",
    help='Access mode for the zarr group that will be accessed. One of "overwrite" (deletes existing data), or "append". Default: "append" (no error if data already exists).',
)
@click.option(
    "--fetch-all-em-resolutions",
    "-all-res",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help='Fetch all resolutions for the EM data. Default: False. Note: setting this to "True" may result in downloading tens or hundreds of GB of data, depending on the crop.',
)
@click.option(
    "-b",
    "--batch-size",
    type=click.INT,
    default=256,
    help="Number of files to fetch in each batch. Default: 256.",
)
@click.option(
    "-w",
    "--num-workers",
    type=click.INT,
    default=int(os.environ.get("CSC_FETCH_DATA_NUM_WORKERS", 32)),
    help=f"Number of workers to use for parallel downloads. Default: {int(os.environ.get('CSC_FETCH_DATA_NUM_WORKERS', 32))}.",
)
@click.option(
    "--zip",
    "use_zip",
    is_flag=True,
    help="Fetch data from a zip file if available.",
)
def fetch_data_cli(
    crops: str,
    raw_padding: int,
    dest: str,
    access_mode: str,
    fetch_all_em_resolutions: bool,
    batch_size: int,
    num_workers: int,
    use_zip: bool,
):
    """
    Download data for the CellMap segmentation challenge.
    """
    if access_mode == "overwrite":
        mode = "w"
    elif access_mode == "append":
        mode = "a"
    else:
        raise ValueError(
            f'Invalid access mode: {access_mode}. Must be one of "overwrite" or "append"'
        )
    fetch_save_start = time.time()
    pool = ThreadPoolExecutor(max_workers=num_workers)
    dest_path_abs = Path(dest).absolute()

    log = structlog.get_logger()

    if use_zip:
        zips_from_manifest = fetch_zip_manifest()

        zip_url = get_zip_if_available(
            crops, raw_padding, fetch_all_em_resolutions, zips_from_manifest
        )
        if zip_url is None:
            p_str = (
                "no raw padding" if raw_padding == 0 else f"raw padding {raw_padding}"
            )
            res_str = (
                "all resolutions"
                if fetch_all_em_resolutions
                else "only the highest resolution"
            )
            log.info(
                f"No zip file found for the requested crops.{str(crops)} with {p_str} and {res_str}."
            )
            log.info("Please rerun the command without the --zip flag.")
            return

        log.info(f"Found a zip file for the requested crops at {zip_url}.")
        zip_path = dest_path_abs / Path(zip_url.name)
        log.info(f"Downloading zip file to {zip_path}")
        download_file_with_progress(str(zip_url), str(zip_path))
        log.info(f"Done after {time.time() - fetch_save_start:0.3f}s")
        log.info("Please unzip the file manually before continuing.")
        log.info("You can use the following command:")
        log.info(f"unzip {zip_path} -d {dest_path_abs}")
        return
    crops_parsed: tuple[CropRow, ...]

    crops_from_manifest = fetch_manifest()

    if crops == "all" or crops == "test":
        test_crops = get_test_crops()
        log.info(f"Found {len(test_crops)} test crops.")

    if crops == "all":
        crops_parsed = crops_from_manifest + test_crops
    elif crops == "test":
        crops_parsed = test_crops
    else:
        crops_split = tuple(int(x) for x in crops.split(","))
        crops_parsed = tuple(filter(lambda v: v.id in crops_split, crops_from_manifest))

    crop_ids = tuple(c.id for c in crops_parsed)

    if len(crops_parsed) == 0:
        log.info(f"No crops found matching {crops}. Doing nothing.")
        return

    log.info(f"Preparing to copy the following crops: {crop_ids}")
    log.info(f"Data will be saved to {dest_path_abs}")

    futures = []
    for crop in crops_parsed:
        log = log.bind(crop_id=crop.id, dataset=crop.dataset)
        em_source_url = crop.em_url

        gt_source_group: None | zarr.Group = None
        if not isinstance(crop.gt_source, TestCropRow):
            gt_source_url = crop.gt_source
            log.info(f"Fetching GT data for crop {crop.id} from {gt_source_url}")
            try:
                gt_source_group = read_group(
                    str(gt_source_url), storage_options={"anon": True}
                )
                log.info(f"Found GT data at {gt_source_url}.")
            except zarr.errors.GroupNotFoundError:
                log.info(
                    f"No Zarr group was found at {gt_source_url}. This crop will be skipped."
                )
                continue
        else:
            gt_source_group = None
            log.info(f"Test crop {crop.id} does not have GT data.")

        em_source_group: None | zarr.Group = None
        try:
            em_source_group = read_group(
                str(em_source_url), storage_options={"anon": True}
            )
            log.info(f"Found EM data at {em_source_url}.")
        except zarr.errors.GroupNotFoundError:
            log.info(
                f"No EM data was found at {em_source_url}. Saving EM data will be skipped."
            )

        dest_root = URL.build(
            scheme="file", path=f"/{dest_path_abs.as_posix().lstrip('/')}"
        ).joinpath(f"{crop.dataset}/{crop.dataset}.zarr")

        gt_dest_path = _resolve_gt_dest_path(crop)
        em_dest_path = _resolve_em_dest_path(crop)

        dest_root_group = zarr.open_group(str(dest_root), mode=mode)
        # create intermediate groups
        dest_root_group.require_group(gt_dest_path)
        dest_crop_group = zarr.open_group(
            str(dest_root / gt_dest_path).replace("%5C", "\\"), mode=mode
        )

        if gt_source_group is None:
            log.info(
                f"No GT data found at any of the possible URLs. No GT data will be fetched for this crop."
            )
        else:
            fs = gt_source_group.store.fs
            store_path = gt_source_group.store.path

            # Using fs.find here is a performance hack until we fix the slowness of traversing the
            # zarr hierarchy to build the list of files

            gt_files = fs.find(store_path)
            crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in gt_files)
            log.info(
                f"Preparing to fetch {len(crop_group_inventory)} files from {gt_source_url}."
            )

            futures.extend(
                partition_copy_store(
                    keys=crop_group_inventory,
                    source_store=gt_source_group.store,
                    dest_store=dest_crop_group.store,
                    batch_size=batch_size,
                    pool=pool,
                )
            )

        if em_source_group is None:
            log.info(
                f"No EM data found at any of the possible URLs. No EM data will be fetched for this crop."
            )
            continue
        else:
            # model the em group locally
            dest_em_group = GroupSpec.from_zarr(em_source_group).to_zarr(
                FSStore(str(dest_root / em_dest_path).replace("%5C", "\\")),
                path="",
                overwrite=(mode == "w"),
            )

            # get the multiscale model of the source em group
            array_wrapper = {"name": "dask_array", "config": {"chunks": "auto"}}
            em_source_arrays = read_multiscale_group(
                em_source_group, array_wrapper=array_wrapper
            )
            padding = raw_padding

            # get the overlapping region between the crop and the full array, in array coordinates.
            em_group_inventory = ()
            em_source_arrays_sorted = sorted(
                em_source_arrays.items(),
                key=lambda kv: np.prod(kv[1].shape),
                reverse=True,
            )
            if isinstance(crop.gt_source, TestCropRow):
                crop_multiscale_group: dict[str, DataArray] | None = None
                base_gt_scale = VectorScale(scale=crop.gt_source.voxel_size)
            else:
                crop_multiscale_group: dict[str, DataArray] | None = None
                for _, group in gt_source_group.groups():
                    try:
                        crop_multiscale_group = read_multiscale_group(
                            group, array_wrapper=array_wrapper
                        )
                        break
                    except (ValueError, TypeError):
                        continue
                if crop_multiscale_group is None:
                    log.info(
                        f"No multiscale groups found in {gt_source_url}. No EM data can be fetched."
                    )
                    continue
                gt_source_arrays_sorted = sorted(
                    crop_multiscale_group.items(),
                    key=lambda kv: np.prod(kv[1].shape),
                    reverse=True,
                )

                # Commented out functionality (also below): apply padding in a resolution-aware way
                # _, (base_em_scale, _) = transforms_from_coords(
                #     em_source_arrays_sorted[0][1].coords, transform_precision=4
                # )
                _, (base_gt_scale, _) = transforms_from_coords(
                    gt_source_arrays_sorted[0][1].coords, transform_precision=4
                )

            for key, array in em_source_arrays_sorted:
                if any(len(coord) <= 1 for coord in array.coords.values()):
                    log.info(
                        f"Skipping scale level {key} because it has no spatial dimensions"
                    )
                    continue
                _, (current_scale, _) = transforms_from_coords(
                    array.coords, transform_precision=4
                )
                if fetch_all_em_resolutions:
                    ratio_threshold = 0
                else:
                    ratio_threshold = 0.9
                scale_ratios = tuple(
                    s_current / s_gt
                    for s_current, s_gt in zip(current_scale.scale, base_gt_scale.scale)
                )
                if all(tuple(x > ratio_threshold for x in scale_ratios)):

                    # # Relative padding based on the scale of the current resolution:
                    # relative_scale = base_em_scale.scale[0] / current_scale.scale[0]
                    # current_pad = int(padding * relative_scale) # Padding relative to the current scale

                    # Uniform voxel padding for all scales:
                    current_pad = padding
                    if isinstance(crop.gt_source, TestCropRow):
                        starts = crop.gt_source.translation
                        stops = tuple(
                            start + size * vs
                            for start, size, vs in zip(
                                starts, crop.gt_source.shape, crop.gt_source.voxel_size
                            )
                        )
                        coords = array.coords.copy()
                        for k, v in zip(array.coords.keys(), np.array((starts, stops))):
                            coords[k] = v
                        slices = subset_to_slice(
                            array,
                            DataArray(
                                dims=array.dims,
                                coords=coords,
                            ),
                        )
                    else:
                        slices = subset_to_slice(array, crop_multiscale_group["s0"])
                    slices_padded = tuple(
                        slice(
                            max(sl.start - current_pad, 0),
                            min(sl.stop + current_pad, shape),
                            sl.step,
                        )
                        for sl, shape in zip(slices, array.shape)
                    )
                    new_chunks = tuple(
                        map(
                            lambda v: f"{key}/{v}",
                            get_chunk_keys(em_source_group[key], slices_padded),
                        )
                    )
                    log.debug(f"Gathering {len(new_chunks)} chunks from level {key}.")
                    em_group_inventory += new_chunks
                else:
                    log.info(
                        f"Skipping scale level {key} because it is sampled more densely than the groundtruth data"
                    )
                em_group_inventory += (f"{key}/.zarray",)
            # em_group_inventory += (".zattrs",)
            log.info(
                f"Preparing to fetch {len(em_group_inventory)} files from {em_source_url}."
            )
            futures.extend(
                partition_copy_store(
                    keys=em_group_inventory,
                    source_store=em_source_group.store,
                    dest_store=dest_em_group.store,
                    batch_size=batch_size,
                    pool=pool,
                )
            )

    log = log.unbind("crop_id", "dataset")
    log = log.bind(save_location=dest_path_abs.path)
    num_iter = len(futures)
    for idx, maybe_result in enumerate(as_completed(futures)):
        try:
            _ = maybe_result.result()
            log.debug(f"Completed fetching batch {idx + 1} / {num_iter}")
        except Exception as e:
            log.exception(e)

    log.unbind("save_location")
    log.info(f"Done after {time.time() - fetch_save_start:0.3f}s")
    log.info(f"Data saved to {dest_path_abs}")
