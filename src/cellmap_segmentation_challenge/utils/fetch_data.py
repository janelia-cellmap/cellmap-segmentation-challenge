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

from .crops import CropRow


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
    return os.path.normpath(f"{crop.alignment}/labels/groundtruth/crop{crop.id}")


def _resolve_em_dest_path(crop: CropRow) -> str:
    return os.path.join(*crop.em_url.parts[crop.em_url.parts.index(crop.alignment) :])


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

        # Normalize Windows-style paths to Unix-style for URL compatibility
        store_path = (
            store.path.split("://")[-1] if "://" in store.path else store.path
        ).replace("\\", "/")

        return URL.build(scheme=protocol, host="", path=f"{store_path}/{path}")
    msg = f"Store with type {type(store)} cannot be resolved to a url"
    raise ValueError(msg)


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
        yield array._chunk_key(cc).rsplit(array.path)[-1].lstrip(os.sep)


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
