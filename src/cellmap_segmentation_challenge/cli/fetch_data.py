from concurrent.futures import ThreadPoolExecutor
import time
import click
from xarray import DataArray
from xarray_ome_ngff import read_multiscale_group
from xarray_ome_ngff.v04.multiscale import transforms_from_coords
from cellmap_segmentation_challenge.utils.fetch_data import (
    _resolve_em_dest_path,
    _resolve_gt_dest_path,
    subset_to_slice,
    partition_copy_store,
    read_group,
    get_chunk_keys,
)
from cellmap_segmentation_challenge.utils.crops import CropRow, fetch_manifest
import structlog
from yarl import URL
import zarr
from pathlib import Path
import numpy as np
from zarr.storage import FSStore
from pydantic_zarr.v2 import GroupSpec
import os

from dotenv import load_dotenv

load_dotenv()

# get constants from environment, falling back to defaults as needed
# manifest_url = os.environ.get("CSC_FETCH_DATA_MANIFEST_URL", None)
manifest_url = "https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/refs/heads/main/src/cellmap_segmentation_challenge/utils/manifest.csv"
if manifest_url is None:
    raise ValueError("No manifest url provided. Quitting.")
num_workers = int(os.environ.get("CSC_FETCH_DATA_NUM_WORKERS", 32))


@click.command
@click.option(
    "--crops",
    type=click.STRING,
    required=True,
    default="all",
    help='A comma-separated list of crops to download, e.g., "111,112,113", or "all" to download all crops. Default: "all".',
)
@click.option(
    "--raw-padding",
    type=click.INT,
    default=0,
    help="Padding to apply to raw data, in voxels. Default: 0.",
)
@click.option(
    "--dest",
    type=click.STRING,
    default="./data",
    help="Path to directory where data will be stored.",
)
@click.option(
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
def fetch_data_cli(
    crops: str,
    raw_padding: str,
    dest: str,
    access_mode: str,
    fetch_all_em_resolutions,
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
    crops_parsed: tuple[CropRow, ...]

    crops_from_manifest = fetch_manifest(manifest_url)

    if crops == "all":
        # crops_parsed = CHALLENGE_CROPS
        crops_parsed = crops_from_manifest
    else:
        crops_split = tuple(int(x) for x in crops.split(","))
        crops_parsed = tuple(filter(lambda v: v.id in crops_split, crops_from_manifest))

    crop_ids = tuple(c.id for c in crops_parsed)

    if len(crops_parsed) == 0:
        log.info(f"No crops found matching {crops}. Doing nothing.")
        return

    log.info(f"Preparing to copy the following crops: {crop_ids}")
    log.info(f"Data will be saved to {dest_path_abs}")

    for crop in crops_parsed:
        log = log.bind(crop_id=crop.id, dataset=crop.dataset)
        gt_save_start = time.time()
        # gt_source_url = _resolve_gt_source_url(crop_url, crop)
        gt_source_url = crop.gt_url
        em_source_url = crop.em_url

        try:
            gt_source_group = read_group(str(gt_source_url))
            log.info(f"Found GT data at {gt_source_url}.")
        except zarr.errors.GroupNotFoundError:
            log.info(
                f"No Zarr group was found at {gt_source_url}. This crop will be skipped."
            )
            continue

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

        dest_root = URL.build(scheme="file", path=str(dest_path_abs)).joinpath(
            f"{crop.dataset}/{crop.dataset}.zarr"
        )
        gt_dest_path = _resolve_gt_dest_path(crop)
        em_dest_path = _resolve_em_dest_path(crop)

        dest_root_group = zarr.open_group(str(dest_root), mode=mode)
        # create intermediate groups
        dest_root_group.require_group(gt_dest_path)
        dest_crop_group = zarr.open_group(str(dest_root / gt_dest_path), mode=mode)

        fs = gt_source_group.store.fs
        store_path = gt_source_group.store.path

        # Using fs.find here is a performance hack until we fix the slowness of traversing the
        # zarr hierarchy to build the list of files

        gt_files = fs.find(store_path)
        crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in gt_files)
        log.info(
            f"Preparing to fetch {len(crop_group_inventory)} files from {gt_source_url}."
        )

        partition_copy_store(
            keys=crop_group_inventory,
            source_store=gt_source_group.store,
            dest_store=dest_crop_group.store,
            batch_size=256,
            pool=pool,
            log=log,
        )

        log.info(
            f"Finished saving crop to local directory after {time.time() - gt_save_start:0.3f}s"
        )
        if em_source_group is None:
            log.info(
                f"No EM data found at any of the possible URLs. No EM data will be fetched for this crop."
            )
            continue
        else:
            # model the em group locally
            em_dest_group = GroupSpec.from_zarr(em_source_group).to_zarr(
                FSStore(str(str(dest_root / em_dest_path))),
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
            else:
                em_group_inventory = ()
                em_source_arrays_sorted = sorted(
                    em_source_arrays.items(),
                    key=lambda kv: np.prod(kv[1].shape),
                    reverse=True,
                )
                gt_source_arrays_sorted = sorted(
                    crop_multiscale_group.items(),
                    key=lambda kv: np.prod(kv[1].shape),
                    reverse=True,
                )

                # apply padding in a resolution-aware way
                _, (base_em_scale, _) = transforms_from_coords(
                    em_source_arrays_sorted[0][1].coords, transform_precision=4
                )
                _, (base_gt_scale, _) = transforms_from_coords(
                    gt_source_arrays_sorted[0][1].coords, transform_precision=4
                )

                for key, array in em_source_arrays_sorted:
                    _, (current_scale, _) = transforms_from_coords(
                        array.coords, transform_precision=4
                    )
                    if fetch_all_em_resolutions:
                        ratio_threshold = 0
                    else:
                        ratio_threshold = 0.9
                    scale_ratios = tuple(
                        s_current / s_gt
                        for s_current, s_gt in zip(
                            current_scale.scale, base_gt_scale.scale
                        )
                    )
                    if all(tuple(x > ratio_threshold for x in scale_ratios)):

                        relative_scale = base_em_scale.scale[0] / current_scale.scale[0]
                        current_pad = int(padding * relative_scale)
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
                                lambda v: "/".join([key, v]),
                                get_chunk_keys(em_source_group[key], slices_padded),
                            )
                        )
                        log.debug(
                            f"Gathering {len(new_chunks)} chunks from level {key}."
                        )
                        em_group_inventory += new_chunks
                    else:
                        log.info(
                            f"Skipping scale level {key} because it is sampled more densely than the groundtruth data"
                        )

                log.info(
                    f"Preparing to fetch {len(em_group_inventory)} files from {em_source_url}."
                )
                partition_copy_store(
                    keys=em_group_inventory,
                    source_store=em_source_group.store,
                    dest_store=em_dest_group.store,
                    batch_size=256,
                    pool=pool,
                    log=log,
                )
                # ensure that intermediate groups are present
                dest_root_group.require_group(em_dest_path)

    log = log.unbind("crop_id", "dataset")
    log.info(f"Done after {time.time() - fetch_save_start:0.3f}s")
