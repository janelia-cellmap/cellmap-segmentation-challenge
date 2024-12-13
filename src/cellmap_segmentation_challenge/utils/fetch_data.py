from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Iterable, Sequence
import os

import numpy as np
import structlog
import toolz
import zarr
import zarr.errors
import zarr.indexing
import zarr.storage
from yarl import URL
from zarr._storage.store import Store
import boto3
from tqdm import tqdm
from .crops import CropRow, ZipDatasetRow


def copy_store(*, keys: Iterable[str], source_store: Store, dest_store: Store):
    """
    Iterate over the keys, copying them from the source store to the dest store
    """
    for key in keys:
        dest_store[key] = source_store[key]


def partition_copy_store(
    *,
    keys,
    source_store,
    dest_store,
    batch_size,
    pool: ThreadPoolExecutor,
):

    keys_partitioned = toolz.partition_all(batch_size, keys)
    keys_partitioned = list(keys_partitioned)
    futures = tuple(
        pool.submit(
            copy_store, keys=batch, source_store=source_store, dest_store=dest_store
        )
        for batch in keys_partitioned
    )
    return futures


def _resolve_gt_dest_path(crop: CropRow) -> str:
    return os.path.join(
        *f"{crop.alignment}/labels/groundtruth/crop{crop.id}".split("/")
    )


def _resolve_em_dest_path(crop: CropRow) -> str:
    return os.path.join(*crop.em_url.parts[crop.em_url.parts.index(crop.alignment) :])


def get_url(node: zarr.Group | zarr.Array) -> URL:
    return get_store_url(node.store, node.path)


def get_store_url(store: zarr.storage.BaseStore, path: str):
    if hasattr(store, "path"):
        if hasattr(store, "fs"):
            protocol = (
                store.fs.protocol[0]
                if isinstance(store.fs.protocol, Sequence)
                else store.fs.protocol
            )
        else:
            protocol = "file"

        # fsstore keeps the protocol in the path, but not s3store
        store_path = store.path.split("://")[-1] if "://" in store.path else store.path
        return URL.build(scheme=protocol, host=store_path, path=path)

    msg = f"Store with type {type(store)} cannot be resolved to a url"
    raise ValueError(msg)


def get_fibsem_path(crop_path: str) -> str:
    """
    Get the path to the reconstructed FIB-SEM data used to create the crop.
    Returns a uri that resolves to a zarr group
    """
    return ""


def get_chunk_keys(
    array: zarr.Array, region: tuple[slice, ...] = ()
) -> Generator[str, None, None]:
    """
    Get the keys for all the chunks in a Zarr array as a generator of strings.
    Returns keys relative to the path of the array.

    copied with modifications from https://github.com/janelia-cellmap/fibsem-tools/blob/2ff3326b38e5565d4860fdd50faaf1448afbb6ae/src/fibsem_tools/io/zarr/core.py#L191

    Parameters
    ----------
    array: zarr.core.Array
        The zarr array to get the chunk keys from
    region: tuple[slice, ...]
        The region in the zarr array get chunks keys from. Defaults to `()`, which
        will result in all the chunk keys being returned.
    Returns
    -------
    Generator[str, None, None]

    """
    indexer = zarr.indexing.BasicIndexer(region, array)
    chunk_coords = (idx.chunk_coords for idx in indexer)
    for cc in chunk_coords:
        yield array._chunk_key(cc).rsplit(array.path)[-1].lstrip("/")


def get_array_objects(
    node: zarr.Array, region: tuple[slice, ...] = ()
) -> tuple[str, ...]:
    """
    Get a list of the objects supporting this array.
    These objects may or may not exist in storage.
    """

    array_metadata_key = ".zarray"
    attrs_key = ".zattrs"
    ckeys = get_chunk_keys(node, region=region)

    out = tuple([array_metadata_key, attrs_key, *ckeys])
    return out


def get_group_objects(node: zarr.Group) -> tuple[str, ...]:
    group_metadata_key = ".zgroup"
    attrs_key = ".zattrs"
    results: tuple[str] = (group_metadata_key, attrs_key)

    for name, subnode in node.items():
        if isinstance(subnode, zarr.Array):
            subobjects = get_array_objects(subnode)
        else:
            subobjects = get_group_objects(subnode)
        results += tuple(map(lambda v: "/".join([name, v]), subobjects))
    return results


def read_group(path: str, **kwargs) -> zarr.Group:
    return zarr.open_group(path, mode="r", **kwargs)


def subset_to_slice(outer_array, inner_array) -> tuple[slice, ...]:
    coords_bounds = {k: c[[0, -1]] for k, c in inner_array.coords.items()}
    subregion = outer_array.sel(coords_bounds, "nearest")
    out = ()
    for dim, value in outer_array.coords.items():
        start = np.where(value == subregion.coords[dim][0])[0].take(0)
        stop = np.where(value == subregion.coords[dim][-1])[0].take(0)
        step = 1
        out += (slice(start, stop, step),)
    return out


def resolve_em_url(em_source_root: URL, em_source_paths: list[str]):
    log = structlog.get_logger()
    for em_url_parts in zip(
        (em_source_root,) * len(em_source_paths), em_source_paths, strict=True
    ):
        maybe_em_source_url = em_url_parts[0] / em_url_parts[1]
        log.info(f"Checking for EM data at {maybe_em_source_url}")
        try:
            return read_group(str(maybe_em_source_url), storage_options={"anon": True})
        except zarr.errors.GroupNotFoundError:
            log.info(f"No EM data found at {maybe_em_source_url}")
    raise zarr.errors.GroupNotFoundError(f"No EM data found in {em_source_root}")


def parse_s3_url(s3_url: str) -> (str, str):
    if not s3_url.startswith("s3://"):
        raise ValueError("URL must start with s3://")
    without_prefix = s3_url[len("s3://") :]
    parts = without_prefix.split("/", 1)
    if len(parts) < 2:
        raise ValueError("Invalid S3 URL. Could not split into bucket and key.")
    return parts[0], parts[1]


def download_file_with_progress(s3_url, local_filename):
    

    s3 = boto3.client("s3")
    bucket_name, object_key = parse_s3_url(s3_url)
    response = s3.head_object(Bucket=bucket_name, Key=object_key)
    total_size = response["ContentLength"]

    progress_bar = tqdm(
        total=total_size, unit="B", unit_scale=True, desc=local_filename, ascii=True
    )

    def progress_callback(bytes_transferred):
        progress_bar.update(bytes_transferred)

    s3.download_file(
        bucket_name, object_key, local_filename, Callback=progress_callback
    )
    progress_bar.close()


def get_zip_if_available(
    crops, raw_padding, fetch_all_em_resolutions, zips_from_manifest
):
    if crops != "all":
        return None

    for z in zips_from_manifest:
        if z.all_res == fetch_all_em_resolutions and z.padding == raw_padding:
            return z.url
    return None
